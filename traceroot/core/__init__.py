"""Core types, configuration, and interfaces for TraceRoot."""

from traceroot.core.types import (
    ExecutionPhase,
    FailureFamily,
    ErrorCategory,
    ConfidenceLevel,
    PropagationRelation,
    JudgeType,
    AggregationStrategy,
)
from traceroot.core.config import TraceRootConfig, load_config

__all__ = [
    "ExecutionPhase",
    "FailureFamily",
    "ErrorCategory",
    "ConfidenceLevel",
    "PropagationRelation",
    "JudgeType",
    "AggregationStrategy",
    "TraceRootConfig",
    "load_config",
]
