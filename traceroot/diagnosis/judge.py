"""Judge abstraction, structured assessment schema, and multi-provider implementations."""

from abc import ABC, abstractmethod
import json
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import jinja2
from pydantic import BaseModel, Field

# Ensure .env values take precedence
load_dotenv(override=True)

from traceroot.core.config import JudgeConfig
from traceroot.core.types import (
    ConfidenceLevel,
    ErrorCategory,
    ExecutionPhase,
    FailureFamily,
    JudgeType,
    PropagationRelation,
)
from traceroot.diagnosis.counterfactual import CounterfactualVerifier
from traceroot.diagnosis.prefilter import DeterministicPreFilter
from traceroot.diagnosis.prompts import (
    DIAGNOSIS_USER_PROMPT_TEMPLATE,
    SYSTEM_DIAGNOSIS_PROMPT,
)
from traceroot.taxonomy.categories import get_category_definition
from traceroot.trace.pruner import HierarchicalTracePruner
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


class StepAssessment(BaseModel):
    """Detailed diagnosis for an individual trajectory step."""
    step_id: int
    phase: ExecutionPhase
    is_error: bool = False
    error_category: ErrorCategory = ErrorCategory.NO_ERROR
    is_root_cause: bool = False
    relation: PropagationRelation = PropagationRelation.NORMAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = ""
    causal_parent_step: Optional[int] = None

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return ConfidenceLevel.from_score(self.confidence)


class JudgeDiagnosis(BaseModel):
    """Comprehensive failure diagnosis emitted by a judge for a trajectory."""
    trajectory_id: str
    failure_detected: bool = False
    root_cause_step_id: Optional[int] = None
    root_cause_phase: Optional[ExecutionPhase] = None
    root_cause_category: ErrorCategory = ErrorCategory.NO_ERROR
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    summary_rationale: str = ""
    step_assessments: List[StepAssessment] = Field(default_factory=list)
    propagated_step_ids: List[int] = Field(default_factory=list)
    judge_name: str = "default_judge"
    judge_type: JudgeType = JudgeType.MOCK
    model_name: str = "mock"
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_step_assessment(self, step_id: int) -> Optional[StepAssessment]:
        for sa in self.step_assessments:
            if sa.step_id == step_id:
                return sa
        return None


class BaseJudge(ABC):
    """Abstract base class for all TraceRoot failure judges."""

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or JudgeConfig()

    @abstractmethod
    def diagnose(self, trajectory: NormalizedTrajectory) -> JudgeDiagnosis:
        """Analyze a normalized trajectory and produce a structured diagnosis."""
        pass


class MockJudge(BaseJudge):
    """Deterministic, rule-based mock judge for testing, CI/CD, and offline benchmarking."""

    def diagnose(self, trajectory: NormalizedTrajectory) -> JudgeDiagnosis:
        step_assessments: List[StepAssessment] = []
        root_cause_step_id: Optional[int] = None
        root_cause_phase: Optional[ExecutionPhase] = None
        root_cause_category = ErrorCategory.NO_ERROR
        failure_detected = False
        propagated_step_ids: List[int] = []

        # Heuristic rules for deterministic diagnosis
        for idx, step in enumerate(trajectory.steps):
            is_err = False
            cat = ErrorCategory.NO_ERROR
            evidence = "Valid step execution."
            parent_step: Optional[int] = None

            # 1. Check for task understanding errors
            if step.phase == ExecutionPhase.TASK_UNDERSTANDING and step.reasoning_context:
                rc_lower = step.reasoning_context.lower()
                if "instead" in rc_lower or "misunderstood" in rc_lower or "wrong goal" in rc_lower:
                    is_err = True
                    cat = ErrorCategory.MISUNDERSTOOD_TASK
                    evidence = f"Misinterpreted user instructions: {step.reasoning_context}"

            # 2. Check for planning failures
            elif step.phase == ExecutionPhase.PLANNING and step.reasoning_context:
                rc_lower = step.reasoning_context.lower()
                if "invalid" in rc_lower or "missing subgoal" in rc_lower or "omitted" in rc_lower:
                    is_err = True
                    cat = ErrorCategory.INVALID_PLAN
                    evidence = f"Planning omitted necessary subgoals: {step.reasoning_context}"

            # 3. Check for wrong tool selection
            elif step.phase == ExecutionPhase.TOOL_SELECTION and step.tool_name:
                t_lower = step.tool_name.lower()
                prompt_lower = trajectory.task_prompt.lower()
                if "calc" in prompt_lower and "search" in t_lower:
                    is_err = True
                    cat = ErrorCategory.WRONG_TOOL_SELECTED
                    evidence = f"Selected '{step.tool_name}' when computation tool was required."
                elif "delete" in t_lower and "format" in prompt_lower:
                    is_err = True
                    cat = ErrorCategory.WRONG_TOOL_SELECTED
                    evidence = f"Selected inappropriate tool '{step.tool_name}'."

            # 4. Check for tool invocation errors (syntax, missing param, bad API)
            elif step.phase == ExecutionPhase.TOOL_INVOCATION and step.tool_arguments is not None:
                args_str = str(step.tool_arguments).lower()
                if "_raw" in step.tool_arguments or "invalid_json" in args_str:
                    is_err = True
                    cat = ErrorCategory.MALFORMED_ARGUMENTS
                    evidence = f"Tool arguments were syntactically malformed: {step.raw_tool_arguments}"
                elif "missing" in str(step.reasoning_context or "").lower():
                    is_err = True
                    cat = ErrorCategory.MISSING_REQUIRED_PARAMETER
                    evidence = "Omitted mandatory tool parameter."
                elif "later than" in str(step.reasoning_context or "").lower() or "earlier" in args_str:
                    is_err = True
                    cat = ErrorCategory.INVALID_API_USAGE
                    evidence = "Semantic argument violation in API call."

            # 5. Check for tool execution errors and unexpected observations
            elif step.phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION:
                # Check for unhandled tool error: previous was tool error, but agent ignores it
                prev_step = trajectory.steps[idx - 1] if idx > 0 else None
                if prev_step and (prev_step.is_tool_error or prev_step.tool_error):
                    if not step.is_tool_error and "success" in str(step.reasoning_context or "").lower():
                        is_err = True
                        cat = ErrorCategory.UNHANDLED_TOOL_ERROR
                        evidence = f"Ignored tool error '{prev_step.tool_error}' and claimed success."
                
                if not is_err and (step.is_tool_error or step.tool_error):
                    is_err = True
                    err_str = str(step.tool_error or step.tool_result or "").lower()
                    if "500" in err_str or "504" in err_str or "timeout" in err_str or "gateway" in err_str:
                        cat = ErrorCategory.TOOL_EXECUTION_ERROR
                    elif "404" in err_str or "bad request" in err_str or "syntax" in err_str or "schema" in err_str:
                        cat = ErrorCategory.INVALID_API_USAGE
                    elif "usernotfound" in err_str or "not found" in err_str:
                        cat = ErrorCategory.UNHANDLED_TOOL_ERROR
                    else:
                        cat = ErrorCategory.TOOL_EXECUTION_ERROR
                    evidence = f"Tool observation returned error: {step.tool_error or step.tool_result}"
                elif not is_err and step.reasoning_context:
                    rc_lower = step.reasoning_context.lower()
                    if "assumed" in rc_lower or "hallucin" in rc_lower:
                        is_err = True
                        cat = ErrorCategory.REASONING_HALLUCINATION
                        evidence = f"Inferred ungrounded assumption: {step.reasoning_context}"

            # 6. Check for recovery failures (loop, premature termination)
            elif step.phase == ExecutionPhase.RECOVERY_ADAPTATION:
                rc_lower = str(step.reasoning_context or "").lower()
                if "retried" in rc_lower or "loop" in rc_lower or "same" in rc_lower:
                    is_err = True
                    cat = ErrorCategory.REPETITIVE_FAILING_ACTION
                    evidence = "Retried identical failing tool call without adapting."
                elif "gave up" in rc_lower or "premature" in rc_lower or "abandoned" in rc_lower:
                    is_err = True
                    cat = ErrorCategory.PREMATURE_TERMINATION
                    evidence = "Terminated early when alternative tools were available."

            # 7. Check for downstream synthesis failures
            elif step.phase == ExecutionPhase.FINAL_SYNTHESIS:
                rc_lower = str(step.reasoning_context or "").lower()
                prompt_lower = trajectory.task_prompt.lower()
                if "json only" in prompt_lower and "{" not in str(trajectory.final_response or ""):
                    is_err = True
                    cat = ErrorCategory.CONSTRAINT_VIOLATION
                    evidence = "Violated output formatting constraint (not JSON)."
                elif "omitted" in rc_lower:
                    is_err = True
                    cat = ErrorCategory.EVIDENCE_OMISSION
                    evidence = "Omitted requested evidence from final response."
                elif ("contradict" in rc_lower or "million" in rc_lower or "$500" in str(trajectory.final_response or "")) and trajectory.task_success is False and not failure_detected:
                    is_err = True
                    cat = ErrorCategory.UNSUPPORTED_CONCLUSION
                    evidence = f"Final synthesis contradicted collected tool evidence: {trajectory.final_response}"
                elif trajectory.task_success is False and not failure_detected:
                    is_err = True
                    cat = ErrorCategory.UNSUPPORTED_CONCLUSION
                    evidence = "Final answer failed to satisfy task requirements."

            # Classify Root Cause vs Propagated
            if is_err:
                if not failure_detected:
                    # First failure = Root Cause
                    failure_detected = True
                    root_cause_step_id = step.step_id
                    root_cause_phase = step.phase
                    root_cause_category = cat
                    relation = PropagationRelation.ROOT_CAUSE
                    is_root = True
                else:
                    propagated_step_ids.append(step.step_id)
                    relation = PropagationRelation.PROPAGATED_CONSEQUENCE
                    is_root = False
                    parent_step = root_cause_step_id
            else:
                if failure_detected:
                    # Step occurring after root cause with contaminated context
                    relation = PropagationRelation.PROPAGATED_CONSEQUENCE
                    parent_step = root_cause_step_id
                else:
                    relation = PropagationRelation.NORMAL
                is_root = False

            step_assessments.append(
                StepAssessment(
                    step_id=step.step_id,
                    phase=step.phase,
                    is_error=is_err,
                    error_category=cat,
                    is_root_cause=is_root,
                    relation=relation,
                    confidence=0.90 if is_err else 0.95,
                    evidence=evidence,
                    causal_parent_step=parent_step,
                )
            )

        summary = (
            f"Failure detected at step {root_cause_step_id} ({root_cause_category.value})"
            if failure_detected
            else "Trajectory executed successfully without failure."
        )

        return JudgeDiagnosis(
            trajectory_id=trajectory.trajectory_id,
            failure_detected=failure_detected,
            root_cause_step_id=root_cause_step_id,
            root_cause_phase=root_cause_phase,
            root_cause_category=root_cause_category,
            confidence=0.92 if failure_detected else 0.98,
            summary_rationale=summary,
            step_assessments=step_assessments,
            propagated_step_ids=propagated_step_ids,
            judge_name=self.config.name,
            judge_type=JudgeType.MOCK,
            model_name=self.config.model,
        )


class HeuristicJudge(BaseJudge):
    """Fast rule-based heuristic judge for rapid detection of overt execution errors."""

    def diagnose(self, trajectory: NormalizedTrajectory) -> JudgeDiagnosis:
        # Re-use MockJudge deterministic logic as baseline
        mock = MockJudge(self.config)
        diagnosis = mock.diagnose(trajectory)
        diagnosis.judge_type = JudgeType.HEURISTIC
        return diagnosis


class LLMJudge(BaseJudge):
    """Production LLM judge supporting Gemini, OpenAI, and Anthropic backends."""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.jinja_env = jinja2.Environment()
        self.prompt_template = self.jinja_env.from_string(
            self.config.prompt_template or DIAGNOSIS_USER_PROMPT_TEMPLATE
        )

    def diagnose(self, trajectory: NormalizedTrajectory) -> JudgeDiagnosis:
        """Analyze a normalized trajectory using hierarchical slicing, LLM reasoning, and counterfactual verification."""
        # Stage 1: Deterministic Pre-Filtering to identify candidate failure windows
        prefilter_result = DeterministicPreFilter.scan(trajectory)

        # Stage 2: Hierarchical Trace Pruning (compress large tool returns outside zoom window)
        pruned_traj, bundle = HierarchicalTracePruner.compress(
            trajectory,
            zoom_window=prefilter_result.zoom_window,
        )

        user_prompt = self.prompt_template.render(
            task_prompt=pruned_traj.task_prompt,
            task_constraints=pruned_traj.task_constraints,
            available_tools=pruned_traj.available_tools,
            steps=pruned_traj.steps,
            final_response=pruned_traj.final_response,
            task_success=pruned_traj.task_success,
        )

        provider = self.config.provider.lower()
        raw_response = ""

        try:
            if provider == "openai":
                raw_response = self._call_openai(user_prompt)
            elif provider == "gemini":
                raw_response = self._call_gemini(user_prompt)
            elif provider == "anthropic":
                raw_response = self._call_anthropic(user_prompt)
            else:
                # Fallback to mock for testing if no live API configured
                diag = MockJudge(self.config).diagnose(trajectory)
                diag.metadata["prefilter_anomalies"] = prefilter_result.has_anomalies
                diag.metadata["compression_ratio"] = bundle.compression_ratio
                return diag

            diagnosis = self._parse_llm_response(raw_response, trajectory)

            # Stage 3: Counterfactual Verification
            if diagnosis.failure_detected and diagnosis.root_cause_step_id:
                cf_result = CounterfactualVerifier.verify_offline(
                    trajectory,
                    diagnosis.root_cause_step_id,
                    diagnosis.root_cause_category,
                )
                if not cf_result.is_confirmed and cf_result.true_root_cause_step:
                    diagnosis.root_cause_step_id = cf_result.true_root_cause_step
                    if cf_result.true_root_cause_category:
                        diagnosis.root_cause_category = cf_result.true_root_cause_category
                    diagnosis.summary_rationale += f" [Causal refinement: {cf_result.verification_rationale}]"

            # Attach prefilter and pruning metadata
            diagnosis.metadata["prefilter_anomalies"] = prefilter_result.has_anomalies
            diagnosis.metadata["compression_ratio"] = bundle.compression_ratio

            return diagnosis

        except Exception as e:
            # If API call fails or key is missing, provide fallback mock diagnosis with warning
            diag = MockJudge(self.config).diagnose(trajectory)
            diag.metadata["llm_error"] = str(e)
            diag.summary_rationale += f" [Fallback due to: {e}]"
            return diag

    def _call_openai(self, user_prompt: str) -> str:
        """Execute diagnosis using OpenAI API."""
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))
        response = client.chat.completions.create(
            model=self.config.model or "gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_DIAGNOSIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    def _call_gemini(self, user_prompt: str) -> str:
        """Execute diagnosis using Google Gemini API with automatic rate-limit backoff."""
        import time

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key.strip() in ("", "dummy_key"):
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not configured. "
                "Please add your free Google AI Studio API key to .env or set the environment variable."
            )

        model_name = self.config.model or "gemini-2.0-flash"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=f"{SYSTEM_DIAGNOSIS_PROMPT}\n\n{user_prompt}",
                        config={"response_mime_type": "application/json"}
                    )
                    return response.text or "{}"
                except Exception as inner_e:
                    # Check if error is related to google-genai import or usage, attempt legacy fallback
                    if "google.genai" in str(inner_e) or "genai" in str(type(inner_e)).lower():
                        import google.generativeai as genai_legacy
                        genai_legacy.configure(api_key=api_key)
                        model = genai_legacy.GenerativeModel(model_name)
                        response = model.generate_content(
                            f"{SYSTEM_DIAGNOSIS_PROMPT}\n\n{user_prompt}",
                            generation_config={"response_mime_type": "application/json"}
                        )
                        return response.text or "{}"
                    raise inner_e

            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg or "rate" in err_msg
                if is_rate_limit and attempt < max_retries - 1:
                    backoff = 2 * (attempt + 1)
                    time.sleep(backoff)
                    continue
                raise e

        return "{}"

    def _call_anthropic(self, user_prompt: str) -> str:
        """Execute diagnosis using Anthropic Claude API."""
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
        response = client.messages.create(
            model=self.config.model or "claude-3-5-sonnet-20241022",
            max_tokens=self.config.max_tokens,
            system=SYSTEM_DIAGNOSIS_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.config.temperature,
        )
        return response.content[0].text if response.content else "{}"

    def _parse_llm_response(self, raw_text: str, trajectory: NormalizedTrajectory) -> JudgeDiagnosis:
        """Extract and validate JSON diagnosis from LLM response."""
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw_text

        data = json.loads(json_str)

        failure_detected = bool(data.get("failure_detected", False))
        root_step_id = data.get("root_cause_step_id")
        root_phase_str = data.get("root_cause_phase")
        root_cat_str = data.get("root_cause_category", ErrorCategory.NO_ERROR.value)

        root_phase = ExecutionPhase.from_str(root_phase_str) if root_phase_str else None
        root_category = ErrorCategory(root_cat_str) if root_cat_str in [c.value for c in ErrorCategory] else ErrorCategory.NO_ERROR

        step_assessments: List[StepAssessment] = []
        raw_assessments = data.get("step_assessments", [])
        propagated_ids: List[int] = []

        step_map = {s.step_id: s for s in trajectory.steps}

        for sa in raw_assessments:
            s_id = int(sa.get("step_id", 1))
            traj_step = step_map.get(s_id)
            phase = traj_step.phase if traj_step else ExecutionPhase.PLANNING
            is_err = bool(sa.get("is_error", False))
            cat_str = sa.get("error_category", ErrorCategory.NO_ERROR.value)
            cat = ErrorCategory(cat_str) if cat_str in [c.value for c in ErrorCategory] else ErrorCategory.NO_ERROR
            is_root = bool(sa.get("is_root_cause", False))
            conf = float(sa.get("confidence", 1.0))
            ev = str(sa.get("evidence", ""))
            parent_id = sa.get("causal_parent_step")

            relation = PropagationRelation.NORMAL
            if is_root:
                relation = PropagationRelation.ROOT_CAUSE
            elif is_err:
                relation = PropagationRelation.PROPAGATED_CONSEQUENCE
                propagated_ids.append(s_id)
            elif failure_detected and root_step_id and s_id > root_step_id:
                relation = PropagationRelation.PROPAGATED_CONSEQUENCE

            step_assessments.append(
                StepAssessment(
                    step_id=s_id,
                    phase=phase,
                    is_error=is_err,
                    error_category=cat,
                    is_root_cause=is_root,
                    relation=relation,
                    confidence=max(0.0, min(1.0, conf)),
                    evidence=ev,
                    causal_parent_step=int(parent_id) if parent_id is not None else None,
                )
            )

        return JudgeDiagnosis(
            trajectory_id=trajectory.trajectory_id,
            failure_detected=failure_detected,
            root_cause_step_id=int(root_step_id) if root_step_id is not None else None,
            root_cause_phase=root_phase,
            root_cause_category=root_category,
            confidence=float(data.get("confidence", 0.9)),
            summary_rationale=str(data.get("summary_rationale", "")),
            step_assessments=step_assessments,
            propagated_step_ids=propagated_ids,
            judge_name=self.config.name,
            judge_type=JudgeType.LLM,
            model_name=self.config.model,
            raw_response=raw_text,
        )
