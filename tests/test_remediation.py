"""Unit tests for Phase 3 automated remediation and code patch generator."""

from traceroot.annotation.seed_dataset import get_seed_dataset
from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.diagnosis.judge import JudgeDiagnosis
from traceroot.diagnosis.remediation import RemediationGenerator
from traceroot.reporting.report import ReportGenerator


def test_remediation_generator_invalid_plan():
    """Test generating prompt diff and guidance for planning failure."""
    trajs, _ = get_seed_dataset()
    traj = trajs[1]  # seed_02_invalid_plan

    diag = JudgeDiagnosis(
        trajectory_id=traj.trajectory_id,
        failure_detected=True,
        root_cause_step_id=2,
        root_cause_phase=ExecutionPhase.PLANNING,
        root_cause_category=ErrorCategory.INVALID_PLAN,
        summary_rationale="Agent created an invalid plan without prerequisite verification.",
    )

    patch = RemediationGenerator.generate(traj, diag)
    assert patch is not None
    assert patch.root_cause_step_id == 2
    assert patch.error_category == ErrorCategory.INVALID_PLAN
    assert "+++ SYSTEM_PROMPT.md" in patch.prompt_diff
    assert "PLANNING RULES" in patch.prompt_diff
    assert len(patch.few_shot_exemplar) > 0


def test_remediation_generator_invalid_api_usage():
    """Test generating Pydantic model validator for invalid API usage."""
    trajs, _ = get_seed_dataset()
    traj = trajs[3]  # seed_04_invalid_api_usage

    diag = JudgeDiagnosis(
        trajectory_id=traj.trajectory_id,
        failure_detected=True,
        root_cause_step_id=4,
        root_cause_phase=ExecutionPhase.TOOL_INVOCATION,
        root_cause_category=ErrorCategory.INVALID_API_USAGE,
        summary_rationale="Passed checkout date preceding checkin date.",
    )

    patch = RemediationGenerator.generate(traj, diag)
    assert patch is not None
    assert "model_validator" in patch.schema_code_patch
    assert "class" in patch.schema_code_patch


def test_html_report_includes_remediation():
    """Test HTML report renders remediation code patch section."""
    trajs, _ = get_seed_dataset()
    traj = trajs[1]

    diag = JudgeDiagnosis(
        trajectory_id=traj.trajectory_id,
        failure_detected=True,
        root_cause_step_id=2,
        root_cause_phase=ExecutionPhase.PLANNING,
        root_cause_category=ErrorCategory.INVALID_PLAN,
    )

    report_gen = ReportGenerator()
    html = report_gen.generate_html(traj, diag)
    assert "Automated Self-Healing Remediation Patches" in html
    assert "SYSTEM_PROMPT.md" in html
