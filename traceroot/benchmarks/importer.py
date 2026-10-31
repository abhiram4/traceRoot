"""Benchmark log importers for τ-bench, GAIA, and public Inspect AI evaluations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import httpx

from traceroot.core.types import ExecutionPhase
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryMetadata, TrajectoryStep


class BenchmarkImporter:
    """Fetches and normalizes pre-computed trajectory logs from public agent benchmarks."""

    @staticmethod
    def import_tau_bench(
        raw_data: Union[str, Dict[str, Any], Path],
        trajectory_id: Optional[str] = None,
    ) -> NormalizedTrajectory:
        """Parse τ-bench (tau-bench) conversation trajectory into TraceRoot format."""
        if isinstance(raw_data, (str, Path)) and str(raw_data).endswith((".json", ".jsonl")):
            with open(raw_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        traj_id = trajectory_id or data.get("task_id", f"tau_{data.get('user_id', 'sample')}")
        task_prompt = data.get("instruction") or data.get("user_message") or data.get("task", "")
        success = bool(data.get("reward", 0) > 0.5 or data.get("success", False))

        steps: List[TrajectoryStep] = []
        messages = data.get("messages", []) or data.get("trajectory", []) or data.get("turns", [])

        step_counter = 1
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "user" and step_counter == 1:
                steps.append(
                    TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.TASK_UNDERSTANDING,
                        turn_id=1,
                        reasoning_context=f"User Task: {content}",
                    )
                )
                step_counter += 1

            elif role == "assistant":
                # Check for planning/thought
                if content and not tool_calls:
                    steps.append(
                        TrajectoryStep(
                            step_id=step_counter,
                            phase=ExecutionPhase.PLANNING,
                            turn_id=1,
                            reasoning_context=content,
                        )
                    )
                    step_counter += 1

                for tc in tool_calls:
                    func = tc.get("function", {})
                    fn_name = func.get("name", "unknown_tool")
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"_raw": args}

                    # Tool selection
                    steps.append(
                        TrajectoryStep(
                            step_id=step_counter,
                            phase=ExecutionPhase.TOOL_SELECTION,
                            turn_id=1,
                            tool_name=fn_name,
                        )
                    )
                    step_counter += 1

                    # Tool invocation
                    steps.append(
                        TrajectoryStep(
                            step_id=step_counter,
                            phase=ExecutionPhase.TOOL_INVOCATION,
                            turn_id=1,
                            tool_name=fn_name,
                            tool_arguments=args,
                        )
                    )
                    step_counter += 1

            elif role == "tool":
                is_err = "error" in content.lower() or "exception" in content.lower()
                steps.append(
                    TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
                        turn_id=1,
                        tool_result=content,
                        is_tool_error=is_err,
                        tool_error=content if is_err else None,
                    )
                )
                step_counter += 1

        # Add final synthesis if response present
        final_res = data.get("final_response") or (messages[-1].get("content") if messages else "")
        if final_res:
            steps.append(
                TrajectoryStep(
                    step_id=step_counter,
                    phase=ExecutionPhase.FINAL_SYNTHESIS,
                    turn_id=1,
                    reasoning_context=f"Final Answer: {final_res}",
                )
            )

        return NormalizedTrajectory(
            trajectory_id=traj_id,
            task_prompt=task_prompt or "τ-bench Task",
            steps=steps,
            final_response=final_res,
            task_success=success,
            metadata=TrajectoryMetadata(
                benchmark_name="tau-bench",
                task_id=str(traj_id),
                framework="tau_bench",
            ),
        )

    @staticmethod
    def import_gaia(
        raw_data: Union[str, Dict[str, Any], Path],
        trajectory_id: Optional[str] = None,
    ) -> NormalizedTrajectory:
        """Parse GAIA benchmark agent trace into TraceRoot format."""
        if isinstance(raw_data, (str, Path)) and str(raw_data).endswith((".json", ".jsonl")):
            with open(raw_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        traj_id = trajectory_id or data.get("task_id", "gaia_sample")
        task_prompt = data.get("Question") or data.get("question") or data.get("prompt", "")
        ground_truth = data.get("Final answer") or data.get("ground_truth")
        agent_answer = data.get("model_answer") or data.get("final_response") or ""
        success = bool(data.get("is_correct", False) or (str(ground_truth).strip() == str(agent_answer).strip()))

        steps: List[TrajectoryStep] = []
        raw_steps = data.get("steps", []) or data.get("intermediate_steps", [])

        step_counter = 1
        steps.append(
            TrajectoryStep(
                step_id=step_counter,
                phase=ExecutionPhase.TASK_UNDERSTANDING,
                turn_id=1,
                reasoning_context=f"Question: {task_prompt}",
            )
        )
        step_counter += 1

        for s in raw_steps:
            thought = s.get("thought") or s.get("reasoning")
            if thought:
                steps.append(
                    TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.PLANNING,
                        turn_id=1,
                        reasoning_context=thought,
                    )
                )
                step_counter += 1

            action = s.get("action") or s.get("tool")
            action_input = s.get("action_input") or s.get("arguments") or {}
            if action:
                steps.append(
                    TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.TOOL_SELECTION,
                        turn_id=1,
                        tool_name=str(action),
                    )
                )
                step_counter += 1

                steps.append(
                    TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.TOOL_INVOCATION,
                        turn_id=1,
                        tool_name=str(action),
                        tool_arguments=action_input if isinstance(action_input, dict) else {"input": action_input},
                    )
                )
                step_counter += 1

            obs = s.get("observation") or s.get("tool_result")
            if obs:
                is_err = "error" in str(obs).lower() or "not found" in str(obs).lower()
                steps.append(
                    TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
                        turn_id=1,
                        tool_result=obs,
                        is_tool_error=is_err,
                        tool_error=str(obs) if is_err else None,
                    )
                )
                step_counter += 1

        steps.append(
            TrajectoryStep(
                step_id=step_counter,
                phase=ExecutionPhase.FINAL_SYNTHESIS,
                turn_id=1,
                reasoning_context=f"Model Answer: {agent_answer}",
            )
        )

        return NormalizedTrajectory(
            trajectory_id=str(traj_id),
            task_prompt=task_prompt or "GAIA Task",
            steps=steps,
            final_response=str(agent_answer),
            task_success=success,
            ground_truth_answer=str(ground_truth) if ground_truth else None,
            metadata=TrajectoryMetadata(
                benchmark_name="GAIA",
                task_id=str(traj_id),
                framework="langchain",
            ),
        )
