"""Pytest fixtures and mock trace generators for TraceRoot tests."""

import pytest
from traceroot.core.types import ExecutionPhase
from traceroot.trace.schema import (
    NormalizedTrajectory,
    TrajectoryMetadata,
    TrajectoryStep,
)


@pytest.fixture
def sample_normalized_trajectory() -> NormalizedTrajectory:
    """Fixture providing a realistic 7-step normalized trajectory."""
    steps = [
        TrajectoryStep(
            step_id=1,
            phase=ExecutionPhase.TASK_UNDERSTANDING,
            turn_id=1,
            reasoning_context="Goal: Search the customer database for order #9821 and check shipment status.",
        ),
        TrajectoryStep(
            step_id=2,
            phase=ExecutionPhase.PLANNING,
            turn_id=1,
            reasoning_context="Plan: 1) Call search_orders tool with order_id 9821. 2) Check tracking details.",
        ),
        TrajectoryStep(
            step_id=3,
            phase=ExecutionPhase.TOOL_SELECTION,
            turn_id=1,
            tool_name="search_orders",
            tool_arguments={"order_id": 9821},
        ),
        TrajectoryStep(
            step_id=4,
            phase=ExecutionPhase.TOOL_INVOCATION,
            turn_id=1,
            tool_name="search_orders",
            tool_arguments={"order_id": 9821},
            raw_tool_arguments='{"order_id": 9821}',
        ),
        TrajectoryStep(
            step_id=5,
            phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
            turn_id=1,
            tool_name="search_orders",
            tool_result={"status": "shipped", "carrier": "FedEx", "tracking_number": "FX-1029384"},
            is_tool_error=False,
        ),
        TrajectoryStep(
            step_id=6,
            phase=ExecutionPhase.PLANNING,
            turn_id=2,
            reasoning_context="The order has shipped via FedEx. I can now compile the final answer.",
        ),
        TrajectoryStep(
            step_id=7,
            phase=ExecutionPhase.FINAL_SYNTHESIS,
            turn_id=2,
            reasoning_context="Order #9821 has been shipped via FedEx with tracking number FX-1029384.",
        ),
    ]

    return NormalizedTrajectory(
        trajectory_id="sample_traj_001",
        task_prompt="Please look up order #9821 and report its shipment status.",
        task_constraints=["Be concise", "Include tracking number"],
        steps=steps,
        final_response="Order #9821 has been shipped via FedEx with tracking number FX-1029384.",
        task_success=True,
        ground_truth_answer="Shipped via FedEx (FX-1029384)",
        metadata=TrajectoryMetadata(
            benchmark_name="ecommerce_ops",
            task_id="task_9821",
            agent_name="SupportAgent",
            model_name="mock-llm",
            framework="traceroot",
        ),
    )


@pytest.fixture
def sample_inspect_ai_log() -> dict:
    """Fixture providing a raw Inspect AI sample log dictionary."""
    return {
        "eval": {
            "task": "gaia_tool_benchmark",
            "model": "gpt-4o",
        },
        "samples": [
            {
                "id": "gaia_sample_42",
                "input": "Calculate the stock price difference for AAPL between Jan 1 and Feb 1.",
                "target": "$12.45",
                "scores": {
                    "accuracy": {"value": 1.0}
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "Calculate the stock price difference for AAPL between Jan 1 and Feb 1."
                    },
                    {
                        "role": "assistant",
                        "content": "I need to fetch AAPL historical prices for Jan 1 and Feb 1.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_stock_history",
                                    "arguments": '{"ticker": "AAPL", "start": "2024-01-01", "end": "2024-02-01"}'
                                }
                            }
                        ]
                    },
                    {
                        "role": "tool",
                        "content": '{"start_price": 185.00, "end_price": 197.45}'
                    },
                    {
                        "role": "assistant",
                        "content": "The stock price increased by $12.45 from $185.00 to $197.45."
                    }
                ],
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": "The stock price increased by $12.45 from $185.00 to $197.45."
                            }
                        }
                    ]
                }
            }
        ]
    }


@pytest.fixture
def sample_react_trace() -> dict:
    """Fixture providing a ReAct format string transcript."""
    return {
        "task_id": "react_sample_10",
        "task_prompt": "What is the capital of Australia and its population?",
        "raw_text": """
Thought: I should search for the capital of Australia.
Action: search_wiki
Action Input: {"query": "Capital of Australia"}
Observation: Canberra is the capital city of Australia with a population of approx 456,000.
Thought: I have the information needed.
Final Answer: The capital of Australia is Canberra, with an estimated population of 456,000.
"""
    }
