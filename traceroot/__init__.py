"""TraceRoot: Failure Localization and Diagnosis for Tool-Using AI Agents."""

__version__ = "0.1.0"

from traceroot.core.types import (
    ExecutionPhase,
    FailureFamily,
    ErrorCategory,
    ConfidenceLevel,
    PropagationRelation,
)
from traceroot.trace.schema import (
    TrajectoryStep,
    ToolCallRecord,
    ToolResultRecord,
    AgentState,
    NormalizedTrajectory,
    TrajectoryMetadata,
)
from traceroot.trace.normalize import TrajectoryNormalizer
from traceroot.trace.segment import TrajectorySegmenter
from traceroot.execution.trace_capture import TrajectoryTracer, capture_trajectory, diagnose, trace
from traceroot.execution.patch import auto_instrument, patch_anthropic, patch_openai
from traceroot.execution.callbacks import TraceRootCallbackHandler
from traceroot.execution.otel_server import OTelServer

__all__ = [
    "__version__",
    "ExecutionPhase",
    "FailureFamily",
    "ErrorCategory",
    "ConfidenceLevel",
    "PropagationRelation",
    "TrajectoryStep",
    "ToolCallRecord",
    "ToolResultRecord",
    "AgentState",
    "NormalizedTrajectory",
    "TrajectoryMetadata",
    "TrajectoryNormalizer",
    "TrajectorySegmenter",
    "TrajectoryTracer",
    "capture_trajectory",
    "trace",
    "diagnose",
    "auto_instrument",
    "patch_openai",
    "patch_anthropic",
    "TraceRootCallbackHandler",
    "OTelServer",
]

