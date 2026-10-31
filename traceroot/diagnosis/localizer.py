"""Failure localization engine: isolates first observed failure from root-cause failure."""

from typing import List, Optional, Tuple
from traceroot.core.types import ErrorCategory, ExecutionPhase, PropagationRelation
from traceroot.diagnosis.judge import StepAssessment
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


class FailureLocalizer:
    """Localizes the earliest causal root failure in an agent trajectory."""

    @staticmethod
    def identify_first_observed_failure(
        trajectory: NormalizedTrajectory,
        assessments: List[StepAssessment],
    ) -> Optional[int]:
        """Find the earliest step ID where an error is visibly detected."""
        for sa in assessments:
            if sa.is_error or sa.error_category != ErrorCategory.NO_ERROR:
                return sa.step_id
        return None

    @staticmethod
    def trace_root_cause(
        trajectory: NormalizedTrajectory,
        assessments: List[StepAssessment],
    ) -> Tuple[Optional[int], Optional[ExecutionPhase], ErrorCategory]:
        """Traceback from observed failures to the earliest causal root failure."""
        assessment_map = {sa.step_id: sa for sa in assessments}
        first_observed = FailureLocalizer.identify_first_observed_failure(trajectory, assessments)

        if not first_observed:
            return None, None, ErrorCategory.NO_ERROR

        # Explicit root cause marked by judge
        explicit_roots = [sa for sa in assessments if sa.is_root_cause]
        if explicit_roots:
            root_sa = explicit_roots[0]
            return root_sa.step_id, root_sa.phase, root_sa.error_category

        # Otherwise, inspect upstream dependencies before first_observed
        # If first observed is tool invocation or tool selection, check if preceding planning step was flawed
        first_sa = assessment_map.get(first_observed)
        if first_sa and first_sa.phase in {ExecutionPhase.TOOL_SELECTION, ExecutionPhase.TOOL_INVOCATION}:
            # Check previous planning steps
            prev_steps = [s for s in trajectory.steps if s.step_id < first_observed and s.phase == ExecutionPhase.PLANNING]
            if prev_steps:
                last_plan = prev_steps[-1]
                plan_sa = assessment_map.get(last_plan.step_id)
                if plan_sa and (plan_sa.is_error or "invalid" in (last_plan.reasoning_context or "").lower()):
                    return last_plan.step_id, ExecutionPhase.PLANNING, ErrorCategory.INVALID_PLAN

        # Default root cause to the first observed error step
        if first_sa:
            return first_sa.step_id, first_sa.phase, first_sa.error_category

        return first_observed, trajectory.get_step(first_observed).phase if trajectory.get_step(first_observed) else None, ErrorCategory.NO_ERROR
