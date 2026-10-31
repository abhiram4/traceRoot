"""Classification evaluation metrics: Accuracy, Macro F1, Per-Category metrics, and Confusion Matrix."""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from traceroot.core.types import ErrorCategory, FailureFamily


class CategoryMetric(BaseModel):
    """Detailed precision/recall/F1 metrics for an individual taxonomy category."""
    category: ErrorCategory
    precision: float
    recall: float
    f1_score: float
    support: int


class ClassificationReport(BaseModel):
    """Overall classification evaluation report."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    micro_f1: float
    total_samples: int
    family_accuracy: float
    category_metrics: List[CategoryMetric] = Field(default_factory=list)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    category_labels: List[str] = Field(default_factory=list)


class ClassificationEvaluator:
    """Evaluates error taxonomy and failure family classification performance."""

    @staticmethod
    def evaluate(
        y_true: List[ErrorCategory | str],
        y_pred: List[ErrorCategory | str],
    ) -> ClassificationReport:
        """Compute comprehensive taxonomy classification metrics."""
        if not y_true or not y_pred:
            raise ValueError("Input label lists cannot be empty.")
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have the same length.")

        true_strs = [c.value if isinstance(c, ErrorCategory) else str(c) for c in y_true]
        pred_strs = [c.value if isinstance(c, ErrorCategory) else str(c) for c in y_pred]

        unique_labels = sorted(list(set(true_strs + pred_strs)))

        # 1. Overall Metrics
        acc = float(np.mean(np.array(true_strs) == np.array(pred_strs)))
        macro_f1 = float(f1_score(true_strs, pred_strs, labels=unique_labels, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(true_strs, pred_strs, labels=unique_labels, average="weighted", zero_division=0))
        micro_f1 = float(f1_score(true_strs, pred_strs, labels=unique_labels, average="micro", zero_division=0))

        # 2. Per-category metrics
        p, r, f, s = precision_recall_fscore_support(
            true_strs, pred_strs, labels=unique_labels, zero_division=0
        )

        cat_metrics: List[CategoryMetric] = []
        for i, label in enumerate(unique_labels):
            try:
                cat_enum = ErrorCategory(label)
            except ValueError:
                cat_enum = ErrorCategory.NO_ERROR

            cat_metrics.append(
                CategoryMetric(
                    category=cat_enum,
                    precision=round(float(p[i]), 4),
                    recall=round(float(r[i]), 4),
                    f1_score=round(float(f[i]), 4),
                    support=int(s[i]),
                )
            )

        # 3. Family-level Accuracy
        true_families = [ErrorCategory(s).family if s in [c.value for c in ErrorCategory] else FailureFamily.NONE for s in true_strs]
        pred_families = [ErrorCategory(s).family if s in [c.value for c in ErrorCategory] else FailureFamily.NONE for s in pred_strs]
        family_acc = float(np.mean(np.array(true_families) == np.array(pred_families)))

        # 4. Confusion Matrix
        cm = confusion_matrix(true_strs, pred_strs, labels=unique_labels).tolist()

        return ClassificationReport(
            accuracy=round(acc, 4),
            macro_f1=round(macro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
            micro_f1=round(micro_f1, 4),
            total_samples=len(y_true),
            family_accuracy=round(family_acc, 4),
            category_metrics=cat_metrics,
            confusion_matrix=cm,
            category_labels=unique_labels,
        )
