"""Trajectory segmentation and phase transition validation."""

from typing import Any, Dict, List, Optional, Set, Tuple
from traceroot.core.types import ExecutionPhase
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


class TrajectorySegmenter:
    """Refines, validates, and enhances phase segmentation in normalized trajectories."""

    # Valid transitions between execution phases
    VALID_TRANSITIONS: Dict[ExecutionPhase, Set[ExecutionPhase]] = {
        ExecutionPhase.TASK_UNDERSTANDING: {
            ExecutionPhase.PLANNING,
            ExecutionPhase.TOOL_SELECTION,
            ExecutionPhase.FINAL_SYNTHESIS,
        },
        ExecutionPhase.PLANNING: {
            ExecutionPhase.TOOL_SELECTION,
            ExecutionPhase.PLANNING,
            ExecutionPhase.FINAL_SYNTHESIS,
        },
        ExecutionPhase.TOOL_SELECTION: {
            ExecutionPhase.TOOL_INVOCATION,
            ExecutionPhase.PLANNING,
        },
        ExecutionPhase.TOOL_INVOCATION: {
            ExecutionPhase.TOOL_RESULT_INTERPRETATION,
            ExecutionPhase.RECOVERY_ADAPTATION,
        },
        ExecutionPhase.TOOL_RESULT_INTERPRETATION: {
            ExecutionPhase.PLANNING,
            ExecutionPhase.TOOL_SELECTION,
            ExecutionPhase.RECOVERY_ADAPTATION,
            ExecutionPhase.FINAL_SYNTHESIS,
        },
        ExecutionPhase.RECOVERY_ADAPTATION: {
            ExecutionPhase.PLANNING,
            ExecutionPhase.TOOL_SELECTION,
            ExecutionPhase.TOOL_INVOCATION,
            ExecutionPhase.FINAL_SYNTHESIS,
        },
        ExecutionPhase.FINAL_SYNTHESIS: set(),  # Terminal phase
    }

    def __init__(self, infer_implicit_phases: bool = True):
        self.infer_implicit_phases = infer_implicit_phases

    def segment_and_refine(self, trajectory: NormalizedTrajectory) -> NormalizedTrajectory:
        """Enrich and ensure structural completeness of execution phases in a trajectory."""
        if not trajectory.steps:
            return trajectory

        refined_steps: List[TrajectoryStep] = []
        step_id = 1

        # Check if first step is Task Understanding
        first_step = trajectory.steps[0]
        if self.infer_implicit_phases and first_step.phase not in {
            ExecutionPhase.TASK_UNDERSTANDING,
            ExecutionPhase.PLANNING,
        }:
            # Insert implicit task understanding
            refined_steps.append(
                TrajectoryStep(
                    step_id=step_id,
                    phase=ExecutionPhase.TASK_UNDERSTANDING,
                    turn_id=1,
                    implicit=True,
                    reasoning_context=f"Implicit task understanding of: {trajectory.task_prompt[:120]}...",
                )
            )
            step_id += 1

        for i, current_step in enumerate(trajectory.steps):
            # Check if this step follows a tool error and should be tagged as Recovery
            if i > 0:
                prev_step = trajectory.steps[i - 1]
                if (
                    prev_step.phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION
                    and prev_step.is_tool_error
                    and current_step.phase == ExecutionPhase.PLANNING
                ):
                    # Tag this planning step as recovery adaptation
                    current_step.phase = ExecutionPhase.RECOVERY_ADAPTATION

            # Check if implicit planning is needed before a tool selection
            if (
                self.infer_implicit_phases
                and current_step.phase == ExecutionPhase.TOOL_SELECTION
                and (
                    not refined_steps
                    or refined_steps[-1].phase
                    not in {ExecutionPhase.PLANNING, ExecutionPhase.TASK_UNDERSTANDING, ExecutionPhase.RECOVERY_ADAPTATION}
                )
            ):
                refined_steps.append(
                    TrajectoryStep(
                        step_id=step_id,
                        phase=ExecutionPhase.PLANNING,
                        turn_id=current_step.turn_id,
                        implicit=True,
                        reasoning_context=f"Implicit plan to invoke '{current_step.tool_name}'",
                    )
                )
                step_id += 1

            # Clone and re-index step
            step_copy = current_step.model_copy(deep=True)
            step_copy.step_id = step_id
            refined_steps.append(step_copy)
            step_id += 1

        # Check if terminal step is Final Synthesis
        if refined_steps and refined_steps[-1].phase != ExecutionPhase.FINAL_SYNTHESIS:
            if trajectory.final_response:
                refined_steps.append(
                    TrajectoryStep(
                        step_id=step_id,
                        phase=ExecutionPhase.FINAL_SYNTHESIS,
                        turn_id=refined_steps[-1].turn_id + 1,
                        implicit=False,
                        reasoning_context=trajectory.final_response,
                    )
                )

        trajectory.steps = refined_steps
        return trajectory

    def validate_transitions(self, trajectory: NormalizedTrajectory) -> List[Dict[str, Any]]:
        """Identify anomalous or illegal phase transitions across the trajectory."""
        anomalies: List[Dict[str, Any]] = []
        if len(trajectory.steps) < 2:
            return anomalies

        for i in range(len(trajectory.steps) - 1):
            curr = trajectory.steps[i]
            nxt = trajectory.steps[i + 1]

            allowed_next = self.VALID_TRANSITIONS.get(curr.phase, set())
            if nxt.phase not in allowed_next:
                anomalies.append({
                    "from_step_id": curr.step_id,
                    "from_phase": curr.phase.value,
                    "to_step_id": nxt.step_id,
                    "to_phase": nxt.phase.value,
                    "warning": f"Unusual transition from {curr.phase.value} to {nxt.phase.value}",
                })

        return anomalies

    def compute_phase_distribution(self, trajectory: NormalizedTrajectory) -> Dict[str, int]:
        """Compute the frequency count of each execution phase."""
        dist = {p.value: 0 for p in ExecutionPhase}
        for step in trajectory.steps:
            dist[step.phase.value] += 1
        return dist
