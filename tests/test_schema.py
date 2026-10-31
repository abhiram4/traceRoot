"""Unit tests for core types, configuration, and trajectory schema."""

import json
import pytest
from traceroot.core.types import (
    ConfidenceLevel,
    ErrorCategory,
    ExecutionPhase,
    FailureFamily,
    PropagationRelation,
)
from traceroot.core.config import TraceRootConfig, load_config
from traceroot.trace.schema import (
    NormalizedTrajectory,
    TrajectoryMetadata,
    TrajectoryStep,
)


def test_execution_phase_conversion():
    """Test string to ExecutionPhase conversion."""
    assert ExecutionPhase.from_str("task_understanding") == ExecutionPhase.TASK_UNDERSTANDING
    assert ExecutionPhase.from_str("tool-selection") == ExecutionPhase.TOOL_SELECTION
    assert ExecutionPhase.from_str("FINAL_SYNTHESIS") == ExecutionPhase.FINAL_SYNTHESIS
    with pytest.raises(ValueError):
        ExecutionPhase.from_str("invalid_phase")


def test_error_category_families():
    """Test that every error category maps correctly to its failure family."""
    assert ErrorCategory.MISUNDERSTOOD_TASK.family == FailureFamily.COGNITIVE
    assert ErrorCategory.INVALID_PLAN.family == FailureFamily.COGNITIVE
    assert ErrorCategory.WRONG_TOOL_SELECTED.family == FailureFamily.TOOL_INTERACTION
    assert ErrorCategory.MALFORMED_ARGUMENTS.family == FailureFamily.TOOL_INTERACTION
    assert ErrorCategory.TOOL_EXECUTION_ERROR.family == FailureFamily.ENVIRONMENTAL
    assert ErrorCategory.UNHANDLED_TOOL_ERROR.family == FailureFamily.RECOVERY
    assert ErrorCategory.UNSUPPORTED_CONCLUSION.family == FailureFamily.SYNTHESIS
    assert ErrorCategory.NO_ERROR.family == FailureFamily.NONE


def test_confidence_level_mapping():
    """Test confidence float to discrete level mapping."""
    assert ConfidenceLevel.from_score(0.95) == ConfidenceLevel.VERY_HIGH
    assert ConfidenceLevel.from_score(0.80) == ConfidenceLevel.HIGH
    assert ConfidenceLevel.from_score(0.60) == ConfidenceLevel.MEDIUM
    assert ConfidenceLevel.from_score(0.30) == ConfidenceLevel.LOW


def test_trajectory_schema_serialization(sample_normalized_trajectory: NormalizedTrajectory, tmp_path):
    """Test full trajectory serialization to and from JSON."""
    traj = sample_normalized_trajectory
    assert traj.total_steps == 7
    assert len(traj.get_tool_calls()) == 1
    assert traj.get_step(3).phase == ExecutionPhase.TOOL_SELECTION

    json_str = traj.to_json()
    reloaded = NormalizedTrajectory.from_json(json_str)
    assert reloaded.trajectory_id == traj.trajectory_id
    assert len(reloaded.steps) == 7
    assert reloaded.steps[0].phase == ExecutionPhase.TASK_UNDERSTANDING

    file_path = tmp_path / "test_traj.json"
    traj.save_json(file_path)
    loaded_from_disk = NormalizedTrajectory.load_json(file_path)
    assert loaded_from_disk.trajectory_id == traj.trajectory_id


def test_config_loading():
    """Test configuration loading with default and custom settings."""
    cfg = load_config()
    assert cfg.project_name == "TraceRoot"
    assert cfg.ensemble.min_agreement_threshold > 0.0
    assert len(cfg.ensemble.judges) >= 1
