"""Zero-code monkeypatching for automatic agent trace capture across OpenAI and Anthropic SDKs."""

import functools
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from traceroot.core.types import ExecutionPhase
from traceroot.execution.trace_capture import TrajectoryTracer
from traceroot.trace.schema import NormalizedTrajectory

logger = logging.getLogger("traceroot.patch")

_OPENAI_PATCHED = False
_ANTHROPIC_PATCHED = False
_ACTIVE_TRACERS: List[TrajectoryTracer] = []


def get_active_tracer() -> Optional[TrajectoryTracer]:
    """Retrieve the most recently registered active tracer."""
    return _ACTIVE_TRACERS[-1] if _ACTIVE_TRACERS else None


def auto_instrument(
    on_complete: Optional[Callable[[NormalizedTrajectory], None]] = None,
    auto_diagnose: bool = False,
    judge_provider: str = "gemini",
) -> None:
    """Enable zero-code auto-instrumentation for OpenAI and Anthropic client calls.
    
    Example:
        import traceroot
        traceroot.auto_instrument(auto_diagnose=True)
        
        # Any standard OpenAI / Anthropic call is now automatically traced!
        client = openai.OpenAI()
        client.chat.completions.create(...)
    """
    patch_openai(on_complete=on_complete, auto_diagnose=auto_diagnose, judge_provider=judge_provider)
    patch_anthropic(on_complete=on_complete, auto_diagnose=auto_diagnose, judge_provider=judge_provider)


def patch_openai(
    on_complete: Optional[Callable[[NormalizedTrajectory], None]] = None,
    auto_diagnose: bool = False,
    judge_provider: str = "gemini",
) -> None:
    """Monkeypatch OpenAI client completions to automatically record agent trajectory steps."""
    global _OPENAI_PATCHED
    if _OPENAI_PATCHED:
        return

    try:
        import openai
        from openai.resources.chat import completions

        original_create = completions.Completions.create

        @functools.wraps(original_create)
        def wrapped_create(*args, **kwargs):
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "openai-model")
            tools = kwargs.get("tools", [])

            # Extract user prompt from first user message
            user_prompt = "OpenAI Agent Task"
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "user":
                    user_prompt = str(m.get("content", ""))
                    break

            tracer = get_active_tracer()
            created_local = False
            if not tracer:
                tracer = TrajectoryTracer(
                    task_prompt=user_prompt,
                    model_name=model,
                    agent_name="openai_agent",
                )
                _ACTIVE_TRACERS.append(tracer)
                created_local = True

            # Process input messages (e.g. tool results fed back in)
            for m in messages:
                if isinstance(m, dict):
                    role = m.get("role")
                    content = m.get("content")
                    if role == "tool":
                        is_err = "error" in str(content).lower() or "exception" in str(content).lower()
                        tracer.log_tool_result(
                            tool_name=m.get("name", "tool"),
                            result=content,
                            error=str(content) if is_err else None,
                            is_error=is_err,
                        )

            # Call original OpenAI create
            response = original_create(*args, **kwargs)

            try:
                # Extract assistant choice
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    msg = getattr(choice, "message", None)
                    if msg:
                        content = getattr(msg, "content", None)
                        tool_calls = getattr(msg, "tool_calls", None)

                        if content and not tool_calls:
                            tracer.log_thought(content)

                        if tool_calls:
                            for tc in tool_calls:
                                fn = getattr(tc, "function", None)
                                if fn:
                                    fn_name = getattr(fn, "name", "tool")
                                    raw_args = getattr(fn, "arguments", "{}")
                                    try:
                                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                                    except Exception:
                                        parsed_args = {"_raw": raw_args}
                                    tracer.log_tool_call(fn_name, parsed_args)

                        if getattr(choice, "finish_reason", "") == "stop" and content:
                            tracer.log_final_response(content, success=True)
            except Exception as parse_err:
                logger.debug(f"TraceRoot OpenAI parsing note: {parse_err}")

            if created_local and _ACTIVE_TRACERS:
                _ACTIVE_TRACERS.remove(tracer)
                traj = tracer.build_trajectory()
                if on_complete:
                    on_complete(traj)
                if auto_diagnose:
                    from traceroot.execution.trace_capture import diagnose
                    diagnose(traj, judge_provider=judge_provider)

            return response

        completions.Completions.create = wrapped_create
        _OPENAI_PATCHED = True
        logger.info("TraceRoot: Successfully monkeypatched OpenAI client.")
    except ImportError:
        pass


def patch_anthropic(
    on_complete: Optional[Callable[[NormalizedTrajectory], None]] = None,
    auto_diagnose: bool = False,
    judge_provider: str = "gemini",
) -> None:
    """Monkeypatch Anthropic client messages to automatically record agent trajectory steps."""
    global _ANTHROPIC_PATCHED
    if _ANTHROPIC_PATCHED:
        return

    try:
        import anthropic
        from anthropic.resources import messages

        original_create = messages.Messages.create

        @functools.wraps(original_create)
        def wrapped_create(*args, **kwargs):
            input_msgs = kwargs.get("messages", [])
            model = kwargs.get("model", "anthropic-model")

            user_prompt = "Anthropic Agent Task"
            for m in input_msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    user_prompt = str(m.get("content", ""))
                    break

            tracer = get_active_tracer()
            created_local = False
            if not tracer:
                tracer = TrajectoryTracer(
                    task_prompt=user_prompt,
                    model_name=model,
                    agent_name="anthropic_agent",
                )
                _ACTIVE_TRACERS.append(tracer)
                created_local = True

            response = original_create(*args, **kwargs)

            try:
                if hasattr(response, "content") and response.content:
                    for block in response.content:
                        block_type = getattr(block, "type", "")
                        if block_type == "text":
                            text_val = getattr(block, "text", "")
                            if text_val:
                                tracer.log_thought(text_val)
                        elif block_type == "tool_use":
                            tool_name = getattr(block, "name", "tool")
                            tool_input = getattr(block, "input", {})
                            tracer.log_tool_call(tool_name, tool_input if isinstance(tool_input, dict) else {"input": tool_input})

                    if getattr(response, "stop_reason", "") == "end_turn":
                        text_parts = [getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text"]
                        if text_parts:
                            tracer.log_final_response(" ".join(text_parts), success=True)
            except Exception as parse_err:
                logger.debug(f"TraceRoot Anthropic parsing note: {parse_err}")

            if created_local and _ACTIVE_TRACERS:
                _ACTIVE_TRACERS.remove(tracer)
                traj = tracer.build_trajectory()
                if on_complete:
                    on_complete(traj)
                if auto_diagnose:
                    from traceroot.execution.trace_capture import diagnose
                    diagnose(traj, judge_provider=judge_provider)

            return response

        messages.Messages.create = wrapped_create
        _ANTHROPIC_PATCHED = True
        logger.info("TraceRoot: Successfully monkeypatched Anthropic client.")
    except ImportError:
        pass
