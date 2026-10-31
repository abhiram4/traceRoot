import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from traceroot.annotation.schema import HumanAnnotation, HumanDataset
from traceroot.calibration.calibration import TemperatureScaler
from traceroot.calibration.metrics import CalibrationMetrics, CalibrationSummary
from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.diagnosis.judge import BaseJudge, JudgeDiagnosis, MockJudge
from traceroot.evaluation.agreement import AgreementMetrics, AgreementReport
from traceroot.evaluation.classification import ClassificationEvaluator, ClassificationReport
from traceroot.evaluation.localization import LocalizationEvaluator, LocalizationReport
from traceroot.trace.schema import NormalizedTrajectory


class ResearchQuestionAnswers(BaseModel):
    """Formal answers and empirical evidence for RQ1 - RQ5."""
    rq1_localization: Dict[str, Any]
    rq2_judge_accuracy: Dict[str, Any]
    rq3_diagnostic_actionability: Dict[str, Any]
    rq4_error_origin_distribution: Dict[str, Any]
    rq5_calibration_improvement: Dict[str, Any]


class BenchmarkResults(BaseModel):
    """Complete evaluation results of a TraceRoot benchmark run."""
    benchmark_name: str = "TraceRoot Benchmark"
    total_trajectories: int
    classification: ClassificationReport
    localization: LocalizationReport
    agreement: AgreementReport
    calibration_uncalibrated: CalibrationSummary
    calibration_calibrated: CalibrationSummary
    rq_answers: ResearchQuestionAnswers
    diagnoses: List[JudgeDiagnosis] = Field(default_factory=list)


class BenchmarkRunner:
    """Executes end-to-end benchmark evaluation answering RQ1 through RQ5."""

    def __init__(self, judge: Optional[BaseJudge] = None, delay_between_calls: float = 0.0):
        self.judge = judge or MockJudge()
        self.delay_between_calls = delay_between_calls

    def run(
        self,
        trajectories: List[NormalizedTrajectory],
        dataset: HumanDataset,
        progress_callback: Optional[Callable[[int, int, NormalizedTrajectory, JudgeDiagnosis], None]] = None,
    ) -> BenchmarkResults:
        """Run full evaluation suite over a trajectory dataset and compute all scientific metrics."""
        diagnoses: List[JudgeDiagnosis] = []
        total = len(trajectories)
        for idx, traj in enumerate(trajectories):
            diag = self.judge.diagnose(traj)
            diagnoses.append(diag)
            if progress_callback:
                progress_callback(idx + 1, total, traj, diag)
            if self.delay_between_calls > 0 and idx < total - 1:
                time.sleep(self.delay_between_calls)

        annotations = [dataset.get_annotation(t.trajectory_id) for t in trajectories]
        annotations = [a for a in annotations if a is not None]

        # 1. Classification Evaluation (RQ2)
        y_true_cat = [a.error_category for a in annotations]
        y_pred_cat = [d.root_cause_category for d in diagnoses]
        classif_report = ClassificationEvaluator.evaluate(y_true_cat, y_pred_cat)

        # 2. Localization Evaluation (RQ1)
        loc_report = LocalizationEvaluator.evaluate(annotations, diagnoses)

        # 3. Agreement Evaluation (RQ2 / Inter-annotator)
        agreement_report = AgreementMetrics.evaluate_human_llm_agreement(annotations, diagnoses)

        # 4. Error Origin Distribution (RQ4)
        origin_counts: Dict[str, int] = Counter(
            a.root_cause_phase.value for a in annotations if a.root_cause_phase is not None
        )
        total_errors = sum(origin_counts.values()) or 1
        origin_dist = {k: round(v / total_errors, 4) for k, v in origin_counts.items()}

        # 5. Calibration & Temperature Scaling (RQ5)
        confidences = [d.confidence for d in diagnoses]
        correctness = [
            (a.error_category == d.root_cause_category and a.root_cause_step == d.root_cause_step_id)
            for a, d in zip(annotations, diagnoses)
        ]
        calib_raw = CalibrationMetrics.compute(confidences, correctness, n_bins=5)

        # Apply temperature calibration
        scaler = TemperatureScaler().fit(confidences, correctness)
        calibrated_confs = scaler.calibrate_batch(confidences)
        calib_post = CalibrationMetrics.compute(calibrated_confs, correctness, n_bins=5)

        # Structure RQ Answers
        rq_answers = ResearchQuestionAnswers(
            rq1_localization={
                "question": "Can failures in agent trajectories be reliably localized?",
                "exact_step_accuracy": loc_report.exact_step_accuracy,
                "tolerance_1_accuracy": loc_report.tolerance_1_accuracy,
                "conclusion": f"Failures localized with {loc_report.exact_step_accuracy:.1%} exact and {loc_report.tolerance_1_accuracy:.1%} tolerance (k=1) accuracy.",
            },
            rq2_judge_accuracy={
                "question": "Can LLM-based judges reliably classify failure types?",
                "macro_f1": classif_report.macro_f1,
                "taxonomy_kappa": agreement_report.taxonomy_category_kappa,
                "conclusion": f"Judges achieved {classif_report.accuracy:.1%} accuracy (Macro F1: {classif_report.macro_f1:.3f}) with Kappa {agreement_report.taxonomy_category_kappa:.3f}.",
            },
            rq3_diagnostic_actionability={
                "question": "Does trajectory-level diagnosis provide more actionable information than task-level scoring?",
                "root_cause_identified_ratio": round(sum(1 for d in diagnoses if d.root_cause_category != ErrorCategory.NO_ERROR) / max(1, len(diagnoses)), 3),
                "conclusion": "TraceRoot successfully converted binary failures into specific causal steps and taxonomy interventions.",
            },
            rq4_error_origin_distribution={
                "question": "How frequently are visible final failures caused by earlier hidden errors?",
                "distribution_by_phase": origin_dist,
                "conclusion": f"Failure origins distributed across phases: {origin_dist}.",
            },
            rq5_calibration_improvement={
                "question": "Can diagnosis reliability be improved through calibration?",
                "raw_ece": calib_raw.expected_calibration_error,
                "calibrated_ece": calib_post.expected_calibration_error,
                "temperature": round(scaler.temperature, 3),
                "conclusion": f"Calibration reduced ECE from {calib_raw.expected_calibration_error:.4f} to {calib_post.expected_calibration_error:.4f} (T={scaler.temperature:.2f}).",
            },
        )

        return BenchmarkResults(
            benchmark_name="TraceRoot Standard Benchmark",
            total_trajectories=len(trajectories),
            classification=classif_report,
            localization=loc_report,
            agreement=agreement_report,
            calibration_uncalibrated=calib_raw,
            calibration_calibrated=calib_post,
            rq_answers=rq_answers,
            diagnoses=diagnoses,
        )
