"""Unit tests for trace normalization and segmentation."""

import pytest
from traceroot.core.types import ExecutionPhase
from traceroot.trace.normalize import TrajectoryNormalizer
from traceroot.trace.segment import TrajectorySegmenter


def test_normalize_inspect_ai(sample_inspect_ai_log: dict):
    """Test normalizing raw Inspect AI log dictionary."""
    normalizer = TrajectoryNormalizer()
    traj = normalizer.normalize(sample_inspect_ai_log)

    assert traj.trajectory_id == "gaia_sample_42"
    assert "stock price" in traj.task_prompt.lower()
    assert traj.task_success is True
    assert traj.metadata.framework == "inspect_ai"
    assert len(traj.steps) >= 4

    phases = [s.phase for s in traj.steps]
    assert ExecutionPhase.TOOL_SELECTION in phases
    assert ExecutionPhase.TOOL_INVOCATION in phases
    assert ExecutionPhase.TOOL_RESULT_INTERPRETATION in phases


def test_normalize_react(sample_react_trace: dict):
    """Test normalizing text-based ReAct agent transcripts."""
    normalizer = TrajectoryNormalizer()
    traj = normalizer.normalize(sample_react_trace)

    assert traj.trajectory_id == "react_sample_10"
    assert len(traj.steps) >= 5
    tool_steps = [s for s in traj.steps if s.phase == ExecutionPhase.TOOL_INVOCATION]
    assert len(tool_steps) == 1
    assert tool_steps[0].tool_name == "search_wiki"
    assert "Canberra" in (traj.final_response or "")


def test_normalize_openai_messages():
    """Test normalizing OpenAI tool calls message array."""
    data = {
        "task_id": "openai_test_1",
        "task_prompt": "Look up weather in Tokyo.",
        "messages": [
            {"role": "user", "content": "Look up weather in Tokyo."},
            {
                "role": "assistant",
                "content": "I'll fetch the Tokyo weather.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}'
                        }
                    }
                ]
            },
            {"role": "tool", "content": '{"temp": 18, "condition": "Sunny"}'},
            {"role": "assistant", "content": "The weather in Tokyo is 18°C and Sunny."}
        ]
    }
    normalizer = TrajectoryNormalizer()
    traj = normalizer.normalize(data)

    assert traj.trajectory_id == "openai_test_1"
    assert len(traj.steps) >= 4
    assert traj.steps[0].phase in {ExecutionPhase.TASK_UNDERSTANDING, ExecutionPhase.PLANNING}


def test_trajectory_segmenter_refinement(sample_react_trace: dict):
    """Test segmenter phase enrichment and implicit step generation."""
    normalizer = TrajectoryNormalizer()
    traj = normalizer.normalize(sample_react_trace)

    segmenter = TrajectorySegmenter(infer_implicit_phases=True)
    refined = segmenter.segment_and_refine(traj)

    assert refined.steps[0].phase == ExecutionPhase.TASK_UNDERSTANDING
    assert refined.steps[-1].phase == ExecutionPhase.FINAL_SYNTHESIS
    
    dist = segmenter.compute_phase_distribution(refined)
    assert dist[ExecutionPhase.TOOL_INVOCATION.value] == 1
    assert dist[ExecutionPhase.FINAL_SYNTHESIS.value] == 1

    anomalies = segmenter.validate_transitions(refined)
    # Refined trajectory should have standard transitions
    assert isinstance(anomalies, list)
