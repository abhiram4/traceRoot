"""Trajectory representation, normalization, and phase segmentation."""

from traceroot.trace.schema import (
    AgentState,
    NormalizedTrajectory,
    ToolCallRecord,
    ToolResultRecord,
    TrajectoryMetadata,
    TrajectoryStep,
)
from traceroot.trace.normalize import TrajectoryNormalizer
from traceroot.trace.segment import TrajectorySegmenter

__all__ = [
    "AgentState",
    "NormalizedTrajectory",
    "ToolCallRecord",
    "ToolResultRecord",
    "TrajectoryMetadata",
    "TrajectoryStep",
    "TrajectoryNormalizer",
    "TrajectorySegmenter",
]
