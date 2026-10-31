"""Normalizer that translates various agent execution traces into the standard TraceRoot schema."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from traceroot.core.types import ExecutionPhase
from traceroot.trace.schema import (
    NormalizedTrajectory,
    TrajectoryMetadata,
    TrajectoryStep,
)


class TrajectoryNormalizer:
    """Normalizes raw agent traces from various frameworks into TraceRoot format."""

    def normalize(
        self,
        raw_trace: Union[Dict[str, Any], str, Path],
        framework: Optional[str] = None,
        task_prompt: Optional[str] = None,
    ) -> NormalizedTrajectory:
        """Main entrypoint: normalizes a raw trace dict or file path."""
        data: Dict[str, Any]
        if isinstance(raw_trace, (str, Path)):
            p = Path(raw_trace)
            if not p.exists():
                raise FileNotFoundError(f"Trace file not found: {raw_trace}")
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = raw_trace

        # Auto-detect framework if not provided
        detected_framework = framework or self._detect_framework(data)

        if detected_framework == "traceroot":
            return NormalizedTrajectory.from_dict(data)
        elif detected_framework == "inspect_ai":
            return self._normalize_inspect_ai(data)
        elif detected_framework == "react":
            return self._normalize_react(data, task_prompt)
        elif detected_framework == "openai_messages":
            return self._normalize_openai_messages(data, task_prompt)
        else:
            # Fallback to generic dictionary parser
            return self._normalize_generic(data, task_prompt)

    def _detect_framework(self, data: Dict[str, Any]) -> str:
        """Heuristically identify the trace source format."""
        if "steps" in data and "trajectory_id" in data and "metadata" in data:
            if isinstance(data.get("steps"), list) and data["steps"] and "phase" in data["steps"][0]:
                return "traceroot"
        if "eval" in data or "samples" in data or "sample_id" in data or "results" in data:
            return "inspect_ai"
        if "messages" in data and isinstance(data["messages"], list):
            return "openai_messages"
        if "trajectory" in data and isinstance(data["trajectory"], str):
            return "react"
        if "raw_text" in data and ("Thought:" in data["raw_text"] or "Action:" in data["raw_text"]):
            return "react"
        return "generic"

    def _normalize_inspect_ai(self, data: Dict[str, Any]) -> NormalizedTrajectory:
        """Normalize Inspect AI evaluation logs or single sample log."""
        # Check if it's a multi-sample inspect log or single sample
        sample = data
        if "samples" in data and isinstance(data["samples"], list) and len(data["samples"]) > 0:
            sample = data["samples"][0]
        
        task_id = str(sample.get("id") or sample.get("sample_id") or "inspect_sample_1")
        task_prompt = ""
        
        # Extract prompt / input
        if "input" in sample:
            inp = sample["input"]
            if isinstance(inp, str):
                task_prompt = inp
            elif isinstance(inp, list):
                task_prompt = "\n".join(str(m.get("content", "")) for m in inp if isinstance(m, dict))
        
        target = str(sample.get("target") or sample.get("ground_truth") or "")
        scores = sample.get("scores", {})
        task_success: Optional[bool] = None
        if scores:
            for _, score_val in scores.items():
                if isinstance(score_val, dict) and "value" in score_val:
                    v = score_val["value"]
                    task_success = (v == 1.0 or v == "C" or v is True or v == "true")
                    break

        metadata = TrajectoryMetadata(
            benchmark_name=str(data.get("eval", {}).get("task") or "inspect_benchmark"),
            task_id=task_id,
            model_name=str(data.get("eval", {}).get("model") or sample.get("model") or "unknown_model"),
            framework="inspect_ai",
            extra={"scores": scores},
        )

        steps: List[TrajectoryStep] = []
        step_counter = 1
        turn_id = 1

        # Extract messages / events
        messages = sample.get("messages", [])
        if not messages and "transcript" in sample:
            transcript = sample["transcript"]
            if isinstance(transcript, dict):
                messages = transcript.get("events", [])

        # Process messages into atomic steps
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "user":
                if not task_prompt:
                    task_prompt = str(content)
                continue

            if role == "assistant":
                # Check for explicit reasoning / thought content
                thought_text = str(content) if content else ""
                
                # If there is thought content
                if thought_text:
                    if step_counter == 1:
                        steps.append(TrajectoryStep(
                            step_id=step_counter,
                            phase=ExecutionPhase.TASK_UNDERSTANDING,
                            turn_id=turn_id,
                            reasoning_context=thought_text,
                        ))
                        step_counter += 1
                    else:
                        steps.append(TrajectoryStep(
                            step_id=step_counter,
                            phase=ExecutionPhase.PLANNING,
                            turn_id=turn_id,
                            reasoning_context=thought_text,
                        ))
                        step_counter += 1

                # If assistant made tool calls
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {}) if isinstance(tc, dict) else {}
                        tool_name = func.get("name") or tc.get("name") or "unknown_tool"
                        raw_args = func.get("arguments") or tc.get("arguments") or {}
                        
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args)
                            except Exception:
                                args = {"_raw": raw_args}
                        else:
                            args = raw_args

                        # Tool Selection step
                        steps.append(TrajectoryStep(
                            step_id=step_counter,
                            phase=ExecutionPhase.TOOL_SELECTION,
                            turn_id=turn_id,
                            tool_name=tool_name,
                            tool_arguments=args,
                        ))
                        step_counter += 1

                        # Tool Invocation step
                        steps.append(TrajectoryStep(
                            step_id=step_counter,
                            phase=ExecutionPhase.TOOL_INVOCATION,
                            turn_id=turn_id,
                            tool_name=tool_name,
                            tool_arguments=args,
                            raw_tool_arguments=str(raw_args),
                        ))
                        step_counter += 1

                elif not tool_calls and thought_text:
                    # Assistant answered without calling tools
                    steps.append(TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.FINAL_SYNTHESIS,
                        turn_id=turn_id,
                        reasoning_context=thought_text,
                    ))
                    step_counter += 1

                turn_id += 1

            elif role == "tool" or role == "observation":
                # Tool result observation
                tool_res = content
                is_err = bool(msg.get("error") or msg.get("is_error"))
                err_msg = str(msg.get("error")) if is_err else None

                # Find the matching tool invocation to link tool name
                last_tool_step = next(
                    (s for s in reversed(steps) if s.phase == ExecutionPhase.TOOL_INVOCATION),
                    None
                )
                t_name = last_tool_step.tool_name if last_tool_step else "unknown_tool"

                steps.append(TrajectoryStep(
                    step_id=step_counter,
                    phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
                    turn_id=turn_id,
                    tool_name=t_name,
                    tool_result=tool_res,
                    tool_error=err_msg,
                    is_tool_error=is_err,
                ))
                step_counter += 1

        final_resp = sample.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content")
        if not final_resp:
            # Fallback to last assistant text
            for step in reversed(steps):
                if step.phase == ExecutionPhase.FINAL_SYNTHESIS:
                    final_resp = step.reasoning_context
                    break

        return NormalizedTrajectory(
            trajectory_id=task_id,
            task_prompt=task_prompt or "Task prompt not specified",
            steps=steps,
            final_response=final_resp,
            task_success=task_success,
            ground_truth_answer=target or None,
            metadata=metadata,
        )

    def _normalize_openai_messages(
        self, data: Dict[str, Any], task_prompt: Optional[str] = None
    ) -> NormalizedTrajectory:
        """Normalize standard OpenAI/Anthropic chat completion message arrays."""
        messages = data.get("messages", [])
        task_id = str(data.get("task_id") or "openai_trace_1")
        prompt = task_prompt or ""
        
        steps: List[TrajectoryStep] = []
        step_counter = 1
        turn_id = 1

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "system":
                continue
            elif role == "user":
                if not prompt:
                    prompt = str(content)
                continue
            elif role == "assistant":
                if content:
                    phase = ExecutionPhase.TASK_UNDERSTANDING if step_counter == 1 else ExecutionPhase.PLANNING
                    if not tool_calls:
                        phase = ExecutionPhase.FINAL_SYNTHESIS
                    steps.append(TrajectoryStep(
                        step_id=step_counter,
                        phase=phase,
                        turn_id=turn_id,
                        reasoning_context=str(content),
                    ))
                    step_counter += 1

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name") or "unknown_tool"
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except Exception:
                        args = {"_raw": args_str}

                    steps.append(TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.TOOL_SELECTION,
                        turn_id=turn_id,
                        tool_name=name,
                        tool_arguments=args,
                    ))
                    step_counter += 1

                    steps.append(TrajectoryStep(
                        step_id=step_counter,
                        phase=ExecutionPhase.TOOL_INVOCATION,
                        turn_id=turn_id,
                        tool_name=name,
                        tool_arguments=args,
                        raw_tool_arguments=str(args_str),
                    ))
                    step_counter += 1
                turn_id += 1

            elif role == "tool":
                last_tool = next(
                    (s for s in reversed(steps) if s.phase == ExecutionPhase.TOOL_INVOCATION),
                    None
                )
                t_name = last_tool.tool_name if last_tool else "tool"
                steps.append(TrajectoryStep(
                    step_id=step_counter,
                    phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
                    turn_id=turn_id,
                    tool_name=t_name,
                    tool_result=content,
                ))
                step_counter += 1

        return NormalizedTrajectory(
            trajectory_id=task_id,
            task_prompt=prompt,
            steps=steps,
            final_response=str(messages[-1].get("content")) if messages and messages[-1].get("role") == "assistant" else None,
            metadata=TrajectoryMetadata(task_id=task_id, framework="openai_messages"),
        )

    def _normalize_react(
        self, data: Dict[str, Any], task_prompt: Optional[str] = None
    ) -> NormalizedTrajectory:
        """Parse text-based ReAct agent transcripts (Thought/Action/Action Input/Observation)."""
        text = str(data.get("raw_text") or data.get("trajectory") or "")
        task_id = str(data.get("task_id") or "react_trace_1")
        prompt = task_prompt or str(data.get("prompt") or data.get("task_prompt") or "")

        steps: List[TrajectoryStep] = []
        step_counter = 1
        turn_id = 1

        # Regex for ReAct blocks
        pattern = re.compile(
            r"(Thought|Action|Action Input|Observation|Final Answer):\s*(.*?)(?=(Thought|Action|Action Input|Observation|Final Answer):|\Z)",
            re.DOTALL | re.IGNORECASE,
        )

        matches = pattern.findall(text)
        current_action_name: Optional[str] = None
        current_action_input: Optional[Dict[str, Any]] = None

        for match in matches:
            block_type = match[0].strip().title()
            content = match[1].strip()

            if block_type == "Thought":
                phase = ExecutionPhase.TASK_UNDERSTANDING if step_counter == 1 else ExecutionPhase.PLANNING
                steps.append(TrajectoryStep(
                    step_id=step_counter,
                    phase=phase,
                    turn_id=turn_id,
                    reasoning_context=content,
                ))
                step_counter += 1

            elif block_type == "Action":
                current_action_name = content
                steps.append(TrajectoryStep(
                    step_id=step_counter,
                    phase=ExecutionPhase.TOOL_SELECTION,
                    turn_id=turn_id,
                    tool_name=content,
                ))
                step_counter += 1

            elif block_type == "Action Input":
                try:
                    args = json.loads(content)
                except Exception:
                    args = {"input": content}
                current_action_input = args

                steps.append(TrajectoryStep(
                    step_id=step_counter,
                    phase=ExecutionPhase.TOOL_INVOCATION,
                    turn_id=turn_id,
                    tool_name=current_action_name or "unknown_tool",
                    tool_arguments=current_action_input,
                    raw_tool_arguments=content,
                ))
                step_counter += 1

            elif block_type == "Observation":
                is_err = "error" in content.lower() or "exception" in content.lower()
                steps.append(TrajectoryStep(
                    step_id=step_counter,
                    phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
                    turn_id=turn_id,
                    tool_name=current_action_name or "unknown_tool",
                    tool_result=content,
                    is_tool_error=is_err,
                ))
                step_counter += 1
                turn_id += 1

            elif block_type == "Final Answer":
                steps.append(TrajectoryStep(
                    step_id=step_counter,
                    phase=ExecutionPhase.FINAL_SYNTHESIS,
                    turn_id=turn_id,
                    reasoning_context=content,
                ))
                step_counter += 1

        final_ans = next(
            (s.reasoning_context for s in reversed(steps) if s.phase == ExecutionPhase.FINAL_SYNTHESIS),
            None
        )

        return NormalizedTrajectory(
            trajectory_id=task_id,
            task_prompt=prompt or "ReAct agent task",
            steps=steps,
            final_response=final_ans,
            metadata=TrajectoryMetadata(task_id=task_id, framework="react"),
        )

    def _normalize_generic(
        self, data: Dict[str, Any], task_prompt: Optional[str] = None
    ) -> NormalizedTrajectory:
        """Generic dictionary normalizer as fallback."""
        task_id = str(data.get("task_id") or data.get("id") or "trace_1")
        prompt = task_prompt or str(data.get("task_prompt") or data.get("prompt") or "")
        raw_steps = data.get("steps", [])

        steps: List[TrajectoryStep] = []
        for idx, s in enumerate(raw_steps, start=1):
            if isinstance(s, dict):
                phase_val = s.get("phase", ExecutionPhase.PLANNING)
                steps.append(TrajectoryStep(
                    step_id=int(s.get("step_id", idx)),
                    phase=ExecutionPhase.from_str(str(phase_val)),
                    turn_id=int(s.get("turn_id", 1)),
                    implicit=bool(s.get("implicit", False)),
                    reasoning_context=s.get("reasoning_context"),
                    tool_name=s.get("tool_name"),
                    tool_arguments=s.get("tool_arguments"),
                    tool_result=s.get("tool_result"),
                    tool_error=s.get("tool_error"),
                    is_tool_error=bool(s.get("is_tool_error", False)),
                ))

        return NormalizedTrajectory(
            trajectory_id=task_id,
            task_prompt=prompt,
            steps=steps,
            final_response=str(data.get("final_response") or ""),
            task_success=data.get("task_success"),
            metadata=TrajectoryMetadata(task_id=task_id, framework="generic"),
        )
