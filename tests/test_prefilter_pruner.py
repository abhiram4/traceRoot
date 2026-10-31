"""Unit tests for Phase 1 Deterministic Pre-Filter, Hierarchical Pruner, and Counterfactual Verifier."""

from traceroot.annotation.seed_dataset import get_seed_dataset
from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.diagnosis.counterfactual import CounterfactualVerifier
from traceroot.diagnosis.judge import LLMJudge
from traceroot.diagnosis.prefilter import DeterministicPreFilter
from traceroot.trace.pruner import HierarchicalTracePruner
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


def test_deterministic_prefilter_malformed_arguments():
    """Test pre-filter detects raw unparsed JSON or malformed arguments."""
    steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Calculate tax"),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="tax_calc", tool_arguments={"_raw": "{invalid_json: true"}),
    ]
    traj = NormalizedTrajectory(trajectory_id="t_malformed", task_prompt="Calc tax", steps=steps)
    
    res = DeterministicPreFilter.scan(traj)
    assert res.has_anomalies is True
    assert res.primary_candidate_step == 2
    assert res.primary_candidate_category == ErrorCategory.MALFORMED_ARGUMENTS


def test_deterministic_prefilter_server_500_error():
    """Test pre-filter detects tool execution errors (500 server crash)."""
    steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Search items"),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="search", tool_arguments={"q": "shoes"}),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_error="HTTP 500 Internal Server Error", is_tool_error=True),
    ]
    traj = NormalizedTrajectory(trajectory_id="t_500", task_prompt="Search items", steps=steps)
    
    res = DeterministicPreFilter.scan(traj)
    assert res.has_anomalies is True
    assert res.primary_candidate_step == 3
    assert res.primary_candidate_category == ErrorCategory.TOOL_EXECUTION_ERROR


def test_deterministic_prefilter_400_invalid_api_usage():
    """Test pre-filter maps 400 Bad Request back to previous tool invocation step."""
    steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Book room"),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="book_hotel", tool_arguments={"checkin": "2024-10-10", "checkout": "2024-10-01"}),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_error="Error 400: checkout cannot precede checkin", is_tool_error=True),
    ]
    traj = NormalizedTrajectory(trajectory_id="t_400", task_prompt="Book room", steps=steps)
    
    res = DeterministicPreFilter.scan(traj)
    assert res.has_anomalies is True
    assert res.primary_candidate_step == 2
    assert res.primary_candidate_category == ErrorCategory.INVALID_API_USAGE


def test_deterministic_prefilter_repetitive_loop():
    """Test pre-filter detects repetitive failing actions."""
    steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="retry_tool", tool_arguments={"id": 1}),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_result="No change"),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="retry_tool", tool_arguments={"id": 1}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_result="No change"),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="retry_tool", tool_arguments={"id": 1}),
    ]
    traj = NormalizedTrajectory(trajectory_id="t_loop", task_prompt="Run loop", steps=steps)

    res = DeterministicPreFilter.scan(traj)
    assert res.has_anomalies is True
    assert any(s.candidate_category == ErrorCategory.REPETITIVE_FAILING_ACTION for s in res.signals)


def test_hierarchical_pruner_compression():
    """Test hierarchical pruner truncates long tool outputs outside the zoom window."""
    long_output = "X" * 1500
    steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Step 1"),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_result=long_output),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_result="Short inside zoom"),
    ]
    traj = NormalizedTrajectory(trajectory_id="t_prune", task_prompt="Prune test", steps=steps)
    
    pruned_traj, bundle = HierarchicalTracePruner.compress(traj, zoom_window=(3, 3))
    assert len(bundle.chunks) >= 1
    # Step 2 outside zoom window should be truncated
    assert len(str(pruned_traj.steps[1].tool_result)) < 500
    assert "truncated" in str(pruned_traj.steps[1].tool_result)
    # Step 3 inside zoom window should be intact
    assert pruned_traj.steps[2].tool_result == "Short inside zoom"


def test_counterfactual_verifier_precedence_correction():
    """Test counterfactual verifier corrects downstream symptom to early planning root cause."""
    trajs, _ = get_seed_dataset()
    t2 = trajs[1]  # seed_02_invalid_plan (Planning error at Step 2, tool error at Step 5)
    
    # If initial hypothesis blames Step 5 (symptomatic tool error)
    cf_res = CounterfactualVerifier.verify_offline(t2, candidate_step_id=5, candidate_category=ErrorCategory.TOOL_EXECUTION_ERROR)
    assert cf_res.is_confirmed is False
    assert cf_res.true_root_cause_step == 2
    assert cf_res.true_root_cause_category == ErrorCategory.INVALID_PLAN


def test_llm_judge_with_prefilter_and_pruner():
    """Test LLMJudge end-to-end pipeline with pre-filtering and pruning metadata."""
    trajs, _ = get_seed_dataset()
    t_sample = trajs[0]  # seed_01_success_order
    
    judge = LLMJudge()
    diag = judge.diagnose(t_sample)
    assert diag is not None
    assert "prefilter_anomalies" in diag.metadata
    assert "compression_ratio" in diag.metadata
