"""Unit tests for Phase 2 zero-code auto-instrumentation, LangChain callbacks, and OTel receiver."""

import time
from traceroot.core.types import ExecutionPhase
from traceroot.execution.callbacks import TraceRootCallbackHandler
from traceroot.execution.otel_server import OTelServer, OTelTraceParser
from traceroot.execution.patch import auto_instrument, patch_anthropic, patch_openai


def test_callback_handler_lifecycle():
    """Test TraceRootCallbackHandler captures thought, tool call, tool result, and final response."""
    handler = TraceRootCallbackHandler(task_prompt="Book flight from NYC to LON")

    # 1. Start chain
    handler.on_chain_start(inputs={"input": "Book flight from NYC to LON"})

    # 2. Tool invocation
    handler.on_tool_start(serialized={"name": "search_flights"}, input_str={"origin": "NYC", "dest": "LON"})

    # 3. Tool observation
    handler.on_tool_end(output={"flights": ["BA102", "AA100"]})

    # 4. Chain completion
    handler.on_chain_end(outputs={"output": "Found 2 flights."})

    traj = handler.get_trajectory()
    assert traj is not None
    assert len(traj.steps) >= 3
    assert traj.task_success is True
    assert traj.final_response == "Found 2 flights."


def test_callback_handler_tool_error():
    """Test TraceRootCallbackHandler captures explicit tool errors."""
    handler = TraceRootCallbackHandler(task_prompt="Fetch weather")

    handler.on_chain_start(inputs={"input": "Fetch weather in Paris"})
    handler.on_tool_start(serialized={"name": "weather_api"}, input_str={"city": "Paris"})
    handler.on_tool_error(error=RuntimeError("500 Internal Server Error"), name="weather_api")
    handler.on_chain_error(error=RuntimeError("Agent execution failed"))

    traj = handler.get_trajectory()
    assert traj is not None
    assert traj.task_success is False
    assert any(s.is_tool_error for s in traj.steps)


def test_otlp_span_parser():
    """Test parsing standard OpenTelemetry GenAI span JSON."""
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace_otel_123",
                                "name": "agent_execution",
                                "attributes": [
                                    {"key": "gen_ai.prompt", "value": {"stringValue": "Calculate 15 * 8"}},
                                    {"key": "gen_ai.tool.name", "value": {"stringValue": "calculator"}},
                                    {"key": "gen_ai.tool.arguments", "value": {"stringValue": '{"expr": "15 * 8"}'}},
                                    {"key": "gen_ai.tool.result", "value": {"stringValue": "120"}},
                                    {"key": "gen_ai.completion", "value": {"stringValue": "The result is 120."}},
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    trajectories = OTelTraceParser.parse_otlp_json(payload)
    assert len(trajectories) == 1
    t = trajectories[0]
    assert t.trajectory_id == "trace_otel_123"
    assert t.task_prompt == "Calculate 15 * 8"
    assert len(t.steps) >= 3
    assert t.final_response == "The result is 120."


def test_otel_server_lifecycle():
    """Test background OTel receiver start and shutdown."""
    server = OTelServer(host="127.0.0.1", port=4399)
    server.start(blocking=False)
    time.sleep(0.2)
    assert server.server is not None
    server.stop()


def test_auto_instrument_api():
    """Test auto_instrument registers monkeypatches gracefully."""
    # Should execute without throwing errors even if client instances differ
    auto_instrument()
    patch_openai()
    patch_anthropic()
