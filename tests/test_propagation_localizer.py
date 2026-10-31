"""Unit tests for failure localization and propagation graph generation."""

import pytest
from traceroot.core.types import (
    ErrorCategory,
    ExecutionPhase,
    PropagationEdgeType,
    PropagationRelation,
)
from traceroot.diagnosis.judge import MockJudge, StepAssessment
from traceroot.diagnosis.localizer import FailureLocalizer
from traceroot.diagnosis.propagation import PropagationGraphBuilder
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryMetadata, TrajectoryStep


@pytest.fixture
def failed_trajectory_with_propagation() -> NormalizedTrajectory:
    """A realistic failed trajectory: Step 2 planning error cascades into Step 3 wrong tool and Step 5 wrong answer."""
    steps = [
        TrajectoryStep(
            step_id=1,
            phase=ExecutionPhase.TASK_UNDERSTANDING,
            reasoning_context="User wants the weather in Paris for tomorrow.",
        ),
        TrajectoryStep(
            step_id=2,
            phase=ExecutionPhase.PLANNING,
            reasoning_context="Invalid plan: I will search internal corporate files instead of using weather API.",
        ),
        TrajectoryStep(
            step_id=3,
            phase=ExecutionPhase.TOOL_SELECTION,
            tool_name="search_internal_files",
            tool_arguments={"query": "Paris weather"},
        ),
        TrajectoryStep(
            step_id=4,
            phase=ExecutionPhase.TOOL_INVOCATION,
            tool_name="search_internal_files",
            tool_arguments={"query": "Paris weather"},
        ),
        TrajectoryStep(
            step_id=5,
            phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
            tool_name="search_internal_files",
            tool_result={"files_found": []},
            is_tool_error=False,
        ),
        TrajectoryStep(
            step_id=6,
            phase=ExecutionPhase.FINAL_SYNTHESIS,
            reasoning_context="No files found, therefore Paris has no weather tomorrow.",
        ),
    ]

    return NormalizedTrajectory(
        trajectory_id="failed_traj_paris_1",
        task_prompt="Get the weather for Paris tomorrow.",
        steps=steps,
        task_success=False,
        metadata=TrajectoryMetadata(task_id="paris_weather_1"),
    )


def test_failure_localization(failed_trajectory_with_propagation: NormalizedTrajectory):
    """Test localizer identifies the root cause step vs downstream consequences."""
    traj = failed_trajectory_with_propagation
    judge = MockJudge()
    diagnosis = judge.diagnose(traj)

    assert diagnosis.failure_detected is True
    assert diagnosis.root_cause_step_id == 2
    assert diagnosis.root_cause_category == ErrorCategory.INVALID_PLAN

    # Test localizer helper
    first_obs = FailureLocalizer.identify_first_observed_failure(traj, diagnosis.step_assessments)
    assert first_obs == 2

    root_s, root_p, root_c = FailureLocalizer.trace_root_cause(traj, diagnosis.step_assessments)
    assert root_s == 2
    assert root_p == ExecutionPhase.PLANNING
    assert root_c == ErrorCategory.INVALID_PLAN


def test_propagation_graph_generation(failed_trajectory_with_propagation: NormalizedTrajectory):
    """Test building NetworkX failure propagation DAG, cascade depth, and Mermaid chart."""
    traj = failed_trajectory_with_propagation
    judge = MockJudge()
    diagnosis = judge.diagnose(traj)

    pg = PropagationGraphBuilder.build(traj, diagnosis)

    assert pg.trajectory_id == traj.trajectory_id
    assert pg.root_cause_nodes == [2]
    assert len(pg.propagated_nodes) >= 3
    assert pg.cascade_depth >= 2
    assert pg.blast_radius > 0.5

    # Test serialization
    data = pg.to_dict()
    assert len(data["nodes"]) == len(traj.steps)
    assert len(data["edges"]) >= len(traj.steps) - 1

    # Test Mermaid generation
    mermaid_str = pg.to_mermaid()
    assert "graph TD" in mermaid_str
    assert "ROOT: invalid_plan" in mermaid_str
    assert "classDef rootNode" in mermaid_str
