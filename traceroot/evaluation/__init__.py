"""Evaluation suite: classification metrics, localization accuracy, inter-annotator agreement, and benchmark runner."""

from traceroot.evaluation.classification import (
    CategoryMetric,
    ClassificationEvaluator,
    ClassificationReport,
)
from traceroot.evaluation.localization import (
    LocalizationEvaluator,
    LocalizationReport,
)
from traceroot.evaluation.agreement import (
    AgreementMetrics,
    AgreementReport,
)
from traceroot.evaluation.benchmark import (
    BenchmarkResults,
    BenchmarkRunner,
    ResearchQuestionAnswers,
)

__all__ = [
    "CategoryMetric",
    "ClassificationEvaluator",
    "ClassificationReport",
    "LocalizationEvaluator",
    "LocalizationReport",
    "AgreementMetrics",
    "AgreementReport",
    "BenchmarkResults",
    "BenchmarkRunner",
    "ResearchQuestionAnswers",
]
