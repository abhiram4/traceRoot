"""Localization evaluation metrics: Exact step accuracy, Tolerance accuracy, Phase accuracy, and Cascade overlap."""

from typing import List, Optional, Set
import numpy as np
from pydantic import BaseModel, Field

from traceroot.annotation.schema import HumanAnnotation
from traceroot.core.types import ExecutionPhase
from traceroot.diagnosis.judge import JudgeDiagnosis


class LocalizationReport(BaseModel):
    """Evaluation report on root-cause step localization and error propagation detection."""
    exact_step_accuracy: float
    tolerance_1_accuracy: float
    tolerance_2_accuracy: float
    phase_accuracy: float
    mean_cascade_jaccard: float
    total_evaluated_samples: int
    failed_trajectories_count: int


class LocalizationEvaluator:
    """Evaluates spatial and temporal failure localization accuracy against human annotations."""

    @staticmethod
    def evaluate(
        annotations: List[HumanAnnotation],
        diagnoses: List[JudgeDiagnosis],
    ) -> LocalizationReport:
        """Compute exact and tolerance-based localization accuracy."""
        diag_map = {d.trajectory_id: d for d in diagnoses}

        exact_matches = 0
        tol_1_matches = 0
        tol_2_matches = 0
        phase_matches = 0
        jaccard_scores: List[float] = []
        failed_count = 0
        evaluated_count = 0

        for ann in annotations:
            diag = diag_map.get(ann.trajectory_id)
            if not diag:
                continue

            evaluated_count += 1
            if not ann.task_success and ann.root_cause_step is not None:
                failed_count += 1
                gold_step = ann.root_cause_step
                pred_step = diag.root_cause_step_id

                if pred_step is not None:
                    # 1. Exact localization
                    if pred_step == gold_step:
                        exact_matches += 1

                    # 2. Tolerance localization (k = 1, k = 2)
                    diff = abs(pred_step - gold_step)
                    if diff <= 1:
                        tol_1_matches += 1
                    if diff <= 2:
                        tol_2_matches += 1

                # 3. Phase accuracy
                gold_phase = ann.root_cause_phase
                pred_phase = diag.root_cause_phase
                if gold_phase is not None and pred_phase is not None:
                    if gold_phase == pred_phase:
                        phase_matches += 1

                # 4. Cascade Jaccard overlap
                gold_cascade: Set[int] = set(ann.propagated_errors)
                pred_cascade: Set[int] = set(diag.propagated_step_ids)
                if gold_cascade or pred_cascade:
                    union = gold_cascade | pred_cascade
                    inter = gold_cascade & pred_cascade
                    jaccard_scores.append(len(inter) / len(union))
                else:
                    jaccard_scores.append(1.0)
            elif ann.task_success:
                # Successes do not have a failure step to localize
                pass

        denom = failed_count if failed_count > 0 else (evaluated_count or 1)

        return LocalizationReport(
            exact_step_accuracy=round(min(1.0, exact_matches / denom), 4),
            tolerance_1_accuracy=round(min(1.0, tol_1_matches / denom), 4),
            tolerance_2_accuracy=round(min(1.0, tol_2_matches / denom), 4),
            phase_accuracy=round(min(1.0, phase_matches / denom), 4),
            mean_cascade_jaccard=round(float(np.mean(jaccard_scores)) if jaccard_scores else 1.0, 4),
            total_evaluated_samples=evaluated_count,
            failed_trajectories_count=failed_count,
        )
