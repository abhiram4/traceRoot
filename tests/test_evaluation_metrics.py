"""Unit tests for evaluation suite: classification, localization, agreement, and benchmark runner."""

import pytest
from traceroot.annotation.seed_dataset import get_multi_annotator_seed, get_seed_dataset
from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.diagnosis.judge import MockJudge
from traceroot.evaluation.agreement import AgreementMetrics
from traceroot.evaluation.benchmark import BenchmarkRunner
from traceroot.evaluation.classification import ClassificationEvaluator
from traceroot.evaluation.localization import LocalizationEvaluator


def test_classification_evaluator():
    """Test Accuracy, Macro F1, and Confusion Matrix computation."""
    y_true = [ErrorCategory.INVALID_PLAN, ErrorCategory.WRONG_TOOL_SELECTED, ErrorCategory.NO_ERROR]
    y_pred = [ErrorCategory.INVALID_PLAN, ErrorCategory.WRONG_TOOL_SELECTED, ErrorCategory.NO_ERROR]

    report = ClassificationEvaluator.evaluate(y_true, y_pred)
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.total_samples == 3
    assert len(report.category_metrics) == 3


def test_localization_evaluator():
    """Test exact step and tolerance-based localization accuracy."""
    trajs, dataset = get_seed_dataset()
    judge = MockJudge()
    diagnoses = [judge.diagnose(t) for t in trajs[:10]]

    report = LocalizationEvaluator.evaluate(dataset.records[:10], diagnoses)
    assert report.total_evaluated_samples == 10
    assert 0.0 <= report.exact_step_accuracy <= 1.0
    assert report.tolerance_1_accuracy >= report.exact_step_accuracy


def test_agreement_metrics():
    """Test Cohen's Kappa, Fleiss' Kappa, and Krippendorff's Alpha."""
    r1 = ["A", "B", "A", "C"]
    r2 = ["A", "B", "A", "C"]
    assert AgreementMetrics.cohens_kappa(r1, r2) == 1.0

    ratings = [
        ["A", "A", "A"],
        ["B", "B", "B"],
        ["C", "C", "C"],
    ]
    f_kappa = AgreementMetrics.fleiss_kappa(ratings)
    assert f_kappa == 1.0

    k_alpha = AgreementMetrics.krippendorffs_alpha(ratings, level_of_measurement="nominal")
    assert k_alpha == 1.0


def test_benchmark_runner_full_run():
    """Test end-to-end benchmark execution answering RQ1-RQ5."""
    trajs, dataset = get_seed_dataset()
    runner = BenchmarkRunner(judge=MockJudge())
    results = runner.run(trajs[:12], dataset)

    assert results.total_trajectories == 12
    assert results.classification.accuracy >= 0.0
    assert results.rq_answers.rq1_localization["exact_step_accuracy"] >= 0.0
    assert results.rq_answers.rq2_judge_accuracy["macro_f1"] >= 0.0
    assert "distribution_by_phase" in results.rq_answers.rq4_error_origin_distribution
    assert results.rq_answers.rq5_calibration_improvement["calibrated_ece"] >= 0.0
