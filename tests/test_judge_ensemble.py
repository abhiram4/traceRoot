"""Unit tests for judge execution, ensemble voting, and disagreement analysis."""

import pytest
from traceroot.core.config import EnsembleConfig, JudgeConfig
from traceroot.core.types import (
    AggregationStrategy,
    ErrorCategory,
    ExecutionPhase,
    JudgeType,
)
from traceroot.diagnosis.ensemble import EnsembleAggregator
from traceroot.diagnosis.judge import HeuristicJudge, JudgeDiagnosis, MockJudge, StepAssessment
from traceroot.trace.schema import NormalizedTrajectory


def test_mock_and_heuristic_judges(sample_normalized_trajectory: NormalizedTrajectory):
    """Test MockJudge and HeuristicJudge on a successful trajectory."""
    judge = MockJudge(JudgeConfig(name="test_mock"))
    diag = judge.diagnose(sample_normalized_trajectory)

    assert diag.trajectory_id == sample_normalized_trajectory.trajectory_id
    assert diag.failure_detected is False
    assert diag.root_cause_category == ErrorCategory.NO_ERROR
    assert len(diag.step_assessments) == sample_normalized_trajectory.total_steps

    h_judge = HeuristicJudge(JudgeConfig(name="test_heuristic"))
    h_diag = h_judge.diagnose(sample_normalized_trajectory)
    assert h_diag.judge_type == JudgeType.HEURISTIC
    assert h_diag.failure_detected is False


def test_ensemble_aggregator_consensus():
    """Test ensemble aggregator with agreeing judges."""
    d1 = JudgeDiagnosis(
        trajectory_id="traj_1",
        failure_detected=True,
        root_cause_step_id=3,
        root_cause_phase=ExecutionPhase.TOOL_SELECTION,
        root_cause_category=ErrorCategory.WRONG_TOOL_SELECTED,
        confidence=0.90,
        judge_name="judge_1",
        step_assessments=[
            StepAssessment(
                step_id=3,
                phase=ExecutionPhase.TOOL_SELECTION,
                is_error=True,
                error_category=ErrorCategory.WRONG_TOOL_SELECTED,
                is_root_cause=True,
            )
        ],
    )
    d2 = JudgeDiagnosis(
        trajectory_id="traj_1",
        failure_detected=True,
        root_cause_step_id=3,
        root_cause_phase=ExecutionPhase.TOOL_SELECTION,
        root_cause_category=ErrorCategory.WRONG_TOOL_SELECTED,
        confidence=0.85,
        judge_name="judge_2",
        step_assessments=[
            StepAssessment(
                step_id=3,
                phase=ExecutionPhase.TOOL_SELECTION,
                is_error=True,
                error_category=ErrorCategory.WRONG_TOOL_SELECTED,
                is_root_cause=True,
            )
        ],
    )

    aggregator = EnsembleAggregator(EnsembleConfig(aggregation_strategy=AggregationStrategy.CONFIDENCE_WEIGHTED))
    res = aggregator.aggregate([d1, d2])

    assert res.failure_detected is True
    assert res.root_cause_step_id == 3
    assert res.root_cause_category == ErrorCategory.WRONG_TOOL_SELECTED
    assert res.confidence > 0.8
    assert res.disagreement_report is not None
    assert not res.disagreement_report.has_high_disagreement


def test_ensemble_aggregator_disagreement():
    """Test ensemble aggregator when judges diverge on step and category."""
    d1 = JudgeDiagnosis(
        trajectory_id="traj_2",
        failure_detected=True,
        root_cause_step_id=2,
        root_cause_phase=ExecutionPhase.PLANNING,
        root_cause_category=ErrorCategory.INVALID_PLAN,
        confidence=0.80,
        judge_name="judge_1",
    )
    d2 = JudgeDiagnosis(
        trajectory_id="traj_2",
        failure_detected=True,
        root_cause_step_id=4,
        root_cause_phase=ExecutionPhase.TOOL_INVOCATION,
        root_cause_category=ErrorCategory.MALFORMED_ARGUMENTS,
        confidence=0.75,
        judge_name="judge_2",
    )
    d3 = JudgeDiagnosis(
        trajectory_id="traj_2",
        failure_detected=False,
        root_cause_step_id=None,
        root_cause_category=ErrorCategory.NO_ERROR,
        confidence=0.50,
        judge_name="judge_3",
    )

    aggregator = EnsembleAggregator(EnsembleConfig(min_agreement_threshold=0.7))
    res = aggregator.aggregate([d1, d2, d3])

    assert len(res.hypotheses) == 3
    assert res.disagreement_report.has_high_disagreement is True
    assert len(res.disagreement_report.disagreement_reasons) > 0
