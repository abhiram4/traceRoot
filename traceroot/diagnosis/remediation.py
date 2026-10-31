"""Automated remediation engine that generates copy-pasteable prompt and tool schema code patches."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.diagnosis.judge import JudgeDiagnosis
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


class RemediationPatch(BaseModel):
    """Structured code and prompt remediation patch generated for a diagnosed root cause."""
    root_cause_step_id: int
    error_category: ErrorCategory
    phase: Optional[ExecutionPhase] = None
    
    # 1. System Prompt Patch (Diff format)
    prompt_diff: str = ""
    prompt_guidance: str = ""
    
    # 2. Pydantic / Python Tool Schema Validator
    schema_code_patch: str = ""
    schema_target_tool: Optional[str] = None
    
    # 3. Few-Shot In-Context Constraint
    few_shot_exemplar: str = ""


class RemediationGenerator:
    """Generates actionable, copy-pasteable engineering fixes for diagnosed agent failures."""

    @classmethod
    def generate(
        cls,
        trajectory: NormalizedTrajectory,
        diagnosis: JudgeDiagnosis,
    ) -> Optional[RemediationPatch]:
        """Generate copy-pasteable code and prompt patches for a diagnosed root cause failure."""
        if not diagnosis.failure_detected or diagnosis.root_cause_step_id is None:
            return None

        step_id = diagnosis.root_cause_step_id
        cat = diagnosis.root_cause_category
        phase = diagnosis.root_cause_phase

        steps_map = {s.step_id: s for s in trajectory.steps}
        root_step = steps_map.get(step_id)

        target_tool = root_step.tool_name if root_step else None
        args = root_step.tool_arguments if root_step else {}

        # 1. Generate System Prompt Patch
        prompt_diff, prompt_guidance = cls._build_prompt_patch(cat, root_step, trajectory)

        # 2. Generate Pydantic Schema Validator
        schema_code = cls._build_schema_patch(cat, root_step, target_tool, args)

        # 3. Generate Few-Shot Counter-Exemplar
        few_shot = cls._build_few_shot_exemplar(cat, root_step, trajectory)

        return RemediationPatch(
            root_cause_step_id=step_id,
            error_category=cat,
            phase=phase,
            prompt_diff=prompt_diff,
            prompt_guidance=prompt_guidance,
            schema_code_patch=schema_code,
            schema_target_tool=target_tool,
            few_shot_exemplar=few_shot,
        )

    @classmethod
    def _build_prompt_patch(
        cls,
        category: ErrorCategory,
        root_step: Optional[TrajectoryStep],
        trajectory: NormalizedTrajectory,
    ) -> tuple[str, str]:
        """Generate markdown prompt diff."""
        if category == ErrorCategory.INVALID_PLAN:
            diff = (
                "--- SYSTEM_PROMPT.md (Original)\n"
                "+++ SYSTEM_PROMPT.md (Patched)\n"
                "@@ -1,5 +1,8 @@\n"
                " You are an autonomous agent.\n"
                "+CRITICAL PLANNING RULES:\n"
                "+1. Always verify prerequisite conditions (e.g., auth tokens, user verification, file existence) before executing state mutations.\n"
                "+2. Decompose complex multi-step goals into verifiable subgoals."
            )
            guidance = "Add explicit prerequisite validation constraints to the system prompt."
        elif category in (ErrorCategory.INVALID_API_USAGE, ErrorCategory.MALFORMED_ARGUMENTS):
            diff = (
                "--- SYSTEM_PROMPT.md (Original)\n"
                "+++ SYSTEM_PROMPT.md (Patched)\n"
                "@@ -1,5 +1,8 @@\n"
                " You are an autonomous agent with tool access.\n"
                "+TOOL CALLING CONSTRAINTS:\n"
                "+1. Strictly adhere to parameter value constraints (e.g. checkin_date <= checkout_date, valid ID formatting).\n"
                "+2. Never formulate unverified or contradictory arguments."
            )
            guidance = "Add parameter boundary constraints and format enforcement to the system prompt."
        elif category == ErrorCategory.UNHANDLED_TOOL_ERROR:
            diff = (
                "--- SYSTEM_PROMPT.md (Original)\n"
                "+++ SYSTEM_PROMPT.md (Patched)\n"
                "@@ -1,5 +1,8 @@\n"
                "+ERROR RECOVERY PROTOCOL:\n"
                "+1. If a tool execution returns an error/exception, DO NOT proceed blindly or claim success.\n"
                "+2. Diagnose the error message, adapt parameters, or try an alternative fallback tool."
            )
            guidance = "Inject an explicit error-recovery protocol requiring the agent to address tool exceptions."
        elif category == ErrorCategory.REPETITIVE_FAILING_ACTION:
            diff = (
                "--- SYSTEM_PROMPT.md (Original)\n"
                "+++ SYSTEM_PROMPT.md (Patched)\n"
                "@@ -1,5 +1,8 @@\n"
                "+LOOP PREVENTION:\n"
                "+1. If a tool call fails, DO NOT retry with the identical arguments.\n"
                "+2. Maximum 2 retry attempts per tool before terminating or asking the user for clarification."
            )
            guidance = "Add loop-prevention and maximum retry constraints."
        else:
            diff = (
                "--- SYSTEM_PROMPT.md (Original)\n"
                "+++ SYSTEM_PROMPT.md (Patched)\n"
                f"+[CONSTRAINT]: Guard against {category.value} by validating step preconditions."
            )
            guidance = f"Add defensive constraints against {category.value} to system instructions."

        return diff, guidance

    @classmethod
    def _build_schema_patch(
        cls,
        category: ErrorCategory,
        root_step: Optional[TrajectoryStep],
        tool_name: Optional[str],
        args: Optional[Dict[str, Any]],
    ) -> str:
        """Generate drop-in Pydantic validator code patch."""
        t_name = tool_name or "CustomToolParams"
        schema_name = f"{t_name.title().replace('_', '')}Input"

        if category == ErrorCategory.INVALID_API_USAGE:
            return (
                f"from pydantic import BaseModel, field_validator, model_validator\n\n"
                f"class {schema_name}(BaseModel):\n"
                f"    # TODO: Add your tool fields here\n"
                f"    \n"
                f"    @model_validator(mode='after')\n"
                f"    def validate_semantic_parameters(self) -> '{schema_name}':\n"
                f"        # TraceRoot Auto-Generated Parameter Guard:\n"
                f"        # Prevents invalid parameter order or contradictory inputs\n"
                f"        return self\n"
            )
        elif category == ErrorCategory.MALFORMED_ARGUMENTS:
            return (
                f"from pydantic import BaseModel, Field\n\n"
                f"class {schema_name}(BaseModel):\n"
                f"    # Enforce strict types to prevent raw malformed JSON arguments\n"
                f"    raw_input: Optional[str] = Field(default=None, description='Clean sanitized input string')\n"
            )
        else:
            return (
                f"from pydantic import BaseModel\n\n"
                f"class {schema_name}(BaseModel):\n"
                f"    # TraceRoot Guard: Validate input parameters before tool execution\n"
                f"    pass\n"
            )

    @classmethod
    def _build_few_shot_exemplar(
        cls,
        category: ErrorCategory,
        root_step: Optional[TrajectoryStep],
        trajectory: NormalizedTrajectory,
    ) -> str:
        """Generate in-context counter-exemplar for few-shot prompt injection."""
        prompt = trajectory.task_prompt[:80]
        step_summary = root_step.to_readable_summary() if root_step else "Faulty step"

        return (
            f"=== FEW-SHOT DEMONSTRATION (Negative Exemplar) ===\n"
            f"User Task: \"{prompt}\"\n"
            f"INCORRECT AGENT BEHAVIOR ({category.value}):\n"
            f"  {step_summary}\n"
            f"CORRECT AGENT BEHAVIOR:\n"
            f"  Verify preconditions and format valid arguments before invocation.\n"
        )
