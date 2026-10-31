"""Deterministic programmatic pre-filter for fast candidate failure localization."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


class CandidateAnomalySignal(BaseModel):
    """An individual deterministic heuristic signal detected during pre-filtering."""
    step_id: int
    phase: ExecutionPhase
    candidate_category: ErrorCategory
    confidence: float
    evidence: str
    is_hard_error: bool = False


class PreFilterResult(BaseModel):
    """Aggregated output of the deterministic pre-filtering scan."""
    has_anomalies: bool = False
    candidate_step_ids: List[int] = Field(default_factory=list)
    primary_candidate_step: Optional[int] = None
    primary_candidate_category: Optional[ErrorCategory] = None
    zoom_window: Optional[Tuple[int, int]] = None
    signals: List[CandidateAnomalySignal] = Field(default_factory=list)
    heuristic_rationale: str = ""


class DeterministicPreFilter:
    """Fast, zero-cost static analyzer that pre-screens trajectories before LLM judging.
    
    Identifies hard runtime errors, HTTP status code anomalies, malformed arguments,
    repetitive execution loops, and unhandled tool exceptions.
    """

    @classmethod
    def scan(cls, trajectory: NormalizedTrajectory) -> PreFilterResult:
        """Analyze trajectory steps and return candidate failure locations."""
        signals: List[CandidateAnomalySignal] = []
        recent_tool_calls: List[Tuple[str, str]] = []

        for idx, step in enumerate(trajectory.steps):
            # 1. Check for malformed tool arguments
            if step.phase == ExecutionPhase.TOOL_INVOCATION:
                if step.tool_arguments is not None:
                    args_str = json.dumps(step.tool_arguments) if isinstance(step.tool_arguments, dict) else str(step.tool_arguments)
                    if "_raw" in step.tool_arguments or "invalid_json" in args_str.lower():
                        signals.append(
                            CandidateAnomalySignal(
                                step_id=step.step_id,
                                phase=step.phase,
                                candidate_category=ErrorCategory.MALFORMED_ARGUMENTS,
                                confidence=0.92,
                                evidence=f"Tool invocation contains raw unparsed or malformed arguments: {step.raw_tool_arguments or args_str}",
                                is_hard_error=True,
                            )
                        )
                    # Check for missing required parameters heuristics
                    elif "missing" in str(step.reasoning_context or "").lower():
                        signals.append(
                            CandidateAnomalySignal(
                                step_id=step.step_id,
                                phase=step.phase,
                                candidate_category=ErrorCategory.MISSING_REQUIRED_PARAMETER,
                                confidence=0.85,
                                evidence="Reasoning context notes missing required parameters.",
                                is_hard_error=False,
                            )
                        )

            # 2. Check for tool execution errors and HTTP status anomalies
            if step.phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION:
                obs_str = str(step.tool_error or step.tool_result or "").lower()
                is_err = step.is_tool_error or bool(step.tool_error)

                # Check for explicit tool execution errors (500, timeout, crash)
                if any(code in obs_str for code in ["500", "502", "503", "504", "internal server error", "timeout", "timed out", "crash"]):
                    signals.append(
                        CandidateAnomalySignal(
                            step_id=step.step_id,
                            phase=step.phase,
                            candidate_category=ErrorCategory.TOOL_EXECUTION_ERROR,
                            confidence=0.90,
                            evidence=f"Tool observation returned explicit server/timeout error: {obs_str[:120]}",
                            is_hard_error=True,
                        )
                    )
                # Check for 400 Bad Request / 404 / 422 Invalid API Usage
                elif any(code in obs_str for code in ["400", "404", "422", "bad request", "invalid argument", "schema validation error", "does not exist", "not found"]):
                    # Lookback: did previous step formulate bad arguments?
                    parent_id = trajectory.steps[idx - 1].step_id if idx > 0 else step.step_id
                    signals.append(
                        CandidateAnomalySignal(
                            step_id=parent_id,
                            phase=ExecutionPhase.TOOL_INVOCATION,
                            candidate_category=ErrorCategory.INVALID_API_USAGE,
                            confidence=0.88,
                            evidence=f"Tool returned 4xx client validation error '{obs_str[:120]}', indicating invalid API usage at step {parent_id}.",
                            is_hard_error=True,
                        )
                    )
                # Check for 403 Forbidden / Permission Denied
                elif "permissiondenied" in obs_str or "403" in obs_str or "unauthorized" in obs_str or "forbidden" in obs_str:
                    # Often caused by planning without auth check
                    signals.append(
                        CandidateAnomalySignal(
                            step_id=step.step_id,
                            phase=step.phase,
                            candidate_category=ErrorCategory.TOOL_EXECUTION_ERROR,
                            confidence=0.85,
                            evidence=f"Observation returned unauthorized/permission error: {obs_str[:120]}",
                            is_hard_error=True,
                        )
                    )
                elif is_err:
                    signals.append(
                        CandidateAnomalySignal(
                            step_id=step.step_id,
                            phase=step.phase,
                            candidate_category=ErrorCategory.TOOL_EXECUTION_ERROR,
                            confidence=0.80,
                            evidence=f"Explicit tool error flag set: {obs_str[:120]}",
                            is_hard_error=True,
                        )
                    )

                # Check for unhandled tool error: previous was error, but agent ignores it and claims success
                prev_step = trajectory.steps[idx - 1] if idx > 0 else None
                if prev_step and (prev_step.is_tool_error or prev_step.tool_error):
                    if not step.is_tool_error and "success" in str(step.reasoning_context or "").lower():
                        signals.append(
                            CandidateAnomalySignal(
                                step_id=step.step_id,
                                phase=step.phase,
                                candidate_category=ErrorCategory.UNHANDLED_TOOL_ERROR,
                                confidence=0.85,
                                evidence="Agent ignored previous explicit tool error and proceeded blindly.",
                                is_hard_error=False,
                            )
                        )

            # 3. Check for repetitive failing loops
            if step.tool_name:
                call_signature = (step.tool_name, str(step.tool_arguments or ""))
                if recent_tool_calls.count(call_signature) >= 2:
                    signals.append(
                        CandidateAnomalySignal(
                            step_id=step.step_id,
                            phase=ExecutionPhase.RECOVERY_ADAPTATION,
                            candidate_category=ErrorCategory.REPETITIVE_FAILING_ACTION,
                            confidence=0.90,
                            evidence=f"Identical tool call '{step.tool_name}' repeated 3+ times without adapting.",
                            is_hard_error=True,
                        )
                    )
                recent_tool_calls.append(call_signature)

            # 4. Check for planning anomalies in reasoning context
            if step.phase in (ExecutionPhase.PLANNING, ExecutionPhase.TASK_UNDERSTANDING) and step.reasoning_context:
                rc_lower = step.reasoning_context.lower()
                if "invalid plan" in rc_lower or "missing subgoal" in rc_lower or "without verifying" in rc_lower:
                    signals.append(
                        CandidateAnomalySignal(
                            step_id=step.step_id,
                            phase=step.phase,
                            candidate_category=ErrorCategory.INVALID_PLAN,
                            confidence=0.92,
                            evidence=f"Planning reasoning explicitly notes invalid subgoal decomposition: {step.reasoning_context[:120]}",
                            is_hard_error=False,
                        )
                    )
                elif "misunderstood" in rc_lower or "wrong goal" in rc_lower:
                    signals.append(
                        CandidateAnomalySignal(
                            step_id=step.step_id,
                            phase=step.phase,
                            candidate_category=ErrorCategory.MISUNDERSTOOD_TASK,
                            confidence=0.90,
                            evidence=f"Reasoning context indicates task misunderstanding: {step.reasoning_context[:120]}",
                            is_hard_error=False,
                        )
                    )

        if not signals:
            return PreFilterResult(
                has_anomalies=False,
                candidate_step_ids=[],
                heuristic_rationale="No explicit deterministic anomalies found during pre-filtering.",
            )

        # Sort signals by earliest step_id (Earliest failure is root cause candidate)
        signals.sort(key=lambda s: s.step_id)
        candidate_ids = list(dict.fromkeys(s.step_id for s in signals))
        primary = signals[0]

        # Determine zoom-in window (±2 steps around earliest anomaly)
        total_steps = len(trajectory.steps)
        min_win = max(1, primary.step_id - 2)
        max_win = min(total_steps, primary.step_id + 3)

        return PreFilterResult(
            has_anomalies=True,
            candidate_step_ids=candidate_ids,
            primary_candidate_step=primary.step_id,
            primary_candidate_category=primary.candidate_category,
            zoom_window=(min_win, max_win),
            signals=signals,
            heuristic_rationale=f"Deterministic pre-filter identified {len(signals)} anomaly signal(s). Earliest candidate: Step {primary.step_id} ({primary.candidate_category.value}).",
        )
