"""Unit tests for OpenTelemetry GenAI Semantic Convention exporter and parser."""

import pytest
from traceroot.core.types import ExecutionPhase
from traceroot.trace.otel import OpenTelemetryExporter
from traceroot.trace.schema import NormalizedTrajectory


def test_otel_span_export_and_roundtrip(sample_normalized_trajectory: NormalizedTrajectory):
    """Test exporting trajectory to OTel spans and parsing back."""
    traj = sample_normalized_trajectory
    spans = OpenTelemetryExporter.trajectory_to_spans(traj)

    assert len(spans) == traj.total_steps + 1  # 1 root span + child step spans

    root_span = spans[0]
    assert "gen_ai.system" in root_span["attributes"]
    assert "gen_ai.agent.name" in root_span["attributes"]
    assert root_span["status"]["code"] == "OK"

    tool_span = next(s for s in spans if "search_orders" in s["attributes"].get("gen_ai.tool.name", ""))
    assert tool_span["attributes"]["traceroot.phase"] in {
        ExecutionPhase.TOOL_SELECTION.value,
        ExecutionPhase.TOOL_INVOCATION.value,
    }

    # Roundtrip test: parse spans back to NormalizedTrajectory
    reconstructed = OpenTelemetryExporter.spans_to_trajectory(spans)
    assert reconstructed.trajectory_id == traj.trajectory_id
    assert len(reconstructed.steps) == traj.total_steps
