"""Runtime context manager and decorator for capturing agent execution traces in real-time."""

from datetime import datetime, timezone
import functools
from typing import Any, Callable, Dict, List, Optional
from traceroot.core.types import ExecutionPhase
from traceroot.trace.schema import (
    NormalizedTrajectory,
    TrajectoryMetadata,
    TrajectoryStep,
)


class TrajectoryTracer:
    """Active runtime tracer that records agent execution steps into NormalizedTrajectory."""

    def __init__(
        self,
        task_prompt: str,
        task_id: Optional[str] = None,
        task_constraints: Optional[List[str]] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        agent_name: str = "custom_agent",
        model_name: str = "custom_model",
        framework: str = "traceroot_runtime",
    ):
        self.task_prompt = task_prompt
        self.task_id = task_id or f"traj_{datetime.now(timezone.utc).timestamp():.4f}"
        self.task_constraints = task_constraints or []
        self.available_tools = available_tools or []
        self.agent_name = agent_name
        self.model_name = model_name
        self.framework = framework

        self.steps: List[TrajectoryStep] = []
        self.step_counter = 1
        self.turn_counter = 1
        self.final_response: Optional[str] = None
        self.task_success: Optional[bool] = None

    def __enter__(self) -> "TrajectoryTracer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None and self.task_success is None:
            self.task_success = False

    def log_thought(self, thought: str, phase: ExecutionPhase = ExecutionPhase.PLANNING) -> int:
        """Log an agent reasoning or planning step."""
        step = TrajectoryStep(
            step_id=self.step_counter,
            phase=phase,
            turn_id=self.turn_counter,
            reasoning_context=thought,
        )
        self.steps.append(step)
        s_id = self.step_counter
        self.step_counter += 1
        return s_id

    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> int:
        """Log a tool selection and invocation step."""
        # 1. Tool selection
        self.steps.append(
            TrajectoryStep(
                step_id=self.step_counter,
                phase=ExecutionPhase.TOOL_SELECTION,
                turn_id=self.turn_counter,
                tool_name=tool_name,
                tool_arguments=arguments,
            )
        )
        self.step_counter += 1

        # 2. Tool invocation
        self.steps.append(
            TrajectoryStep(
                step_id=self.step_counter,
                phase=ExecutionPhase.TOOL_INVOCATION,
                turn_id=self.turn_counter,
                tool_name=tool_name,
                tool_arguments=arguments,
            )
        )
        s_id = self.step_counter
        self.step_counter += 1
        return s_id

    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        error: Optional[str] = None,
        is_error: bool = False,
    ) -> int:
        """Log a tool observation result or error."""
        step = TrajectoryStep(
            step_id=self.step_counter,
            phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
            turn_id=self.turn_counter,
            tool_name=tool_name,
            tool_result=result,
            tool_error=error,
            is_tool_error=is_error or bool(error),
        )
        self.steps.append(step)
        s_id = self.step_counter
        self.step_counter += 1
        self.turn_counter += 1
        return s_id

    def log_final_response(self, response: str, success: Optional[bool] = None) -> int:
        """Log the final synthesis response and mark task outcome."""
        step = TrajectoryStep(
            step_id=self.step_counter,
            phase=ExecutionPhase.FINAL_SYNTHESIS,
            turn_id=self.turn_counter,
            reasoning_context=response,
        )
        self.steps.append(step)
        self.final_response = response
        if success is not None:
            self.task_success = success
        s_id = self.step_counter
        self.step_counter += 1
        return s_id

    def build_trajectory(self) -> NormalizedTrajectory:
        """Build and return the complete NormalizedTrajectory."""
        return NormalizedTrajectory(
            trajectory_id=self.task_id,
            task_prompt=self.task_prompt,
            task_constraints=self.task_constraints,
            available_tools=self.available_tools,
            steps=self.steps,
            final_response=self.final_response,
            task_success=self.task_success,
            metadata=TrajectoryMetadata(
                task_id=self.task_id,
                agent_name=self.agent_name,
                model_name=self.model_name,
                framework=self.framework,
            ),
        )


def capture_trajectory(task_prompt: str, **tracer_kwargs) -> Callable:
    """Function decorator that automatically instruments an agent execution function."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with TrajectoryTracer(task_prompt=task_prompt, **tracer_kwargs) as tracer:
                res = fn(*args, tracer=tracer, **kwargs)
                return res, tracer.build_trajectory()
        return wrapper
    return decorator


def trace(task_prompt: Optional[str] = None, **tracer_kwargs) -> Callable:
    """Convenient decorator to automatically capture agent execution trajectory.

    Example:
        @traceroot.trace("Check weather in Seattle")
        def my_agent(tracer: TrajectoryTracer):
            tracer.log_thought("Looking up weather")
            tracer.log_tool_call("weather_api", {"city": "Seattle"})
            tracer.log_tool_result("weather_api", {"temp": 65})
            return "Weather in Seattle is 65F"
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            prompt = (
                task_prompt
                or kwargs.get("task_prompt")
                or (args[0] if args and isinstance(args[0], str) else "Agent Execution")
            )
            with TrajectoryTracer(task_prompt=prompt, **tracer_kwargs) as tracer:
                res = fn(*args, tracer=tracer, **kwargs)
                return res, tracer.build_trajectory()
        return wrapper
    return decorator


def diagnose(
    trajectory_or_path: Any,
    judge_provider: str = "gemini",
    model: str = "gemini-2.0-flash",
) -> Any:
    """One-line diagnosis function for any trajectory object or JSON log path.

    Example:
        from traceroot import diagnose
        diagnosis = diagnose("path/to/trace.json", judge_provider="gemini")
        print(diagnosis.summary_rationale)
    """
    from traceroot.core.config import JudgeConfig
    from traceroot.core.types import JudgeType
    from traceroot.diagnosis.judge import LLMJudge, MockJudge
    from traceroot.trace.normalize import TrajectoryNormalizer

    if isinstance(trajectory_or_path, NormalizedTrajectory):
        traj = trajectory_or_path
    else:
        normalizer = TrajectoryNormalizer()
        traj = normalizer.normalize(trajectory_or_path)

    j_cfg = JudgeConfig(
        name="runtime_judge",
        judge_type=JudgeType.LLM if judge_provider != "mock" else JudgeType.MOCK,
        provider=judge_provider,
        model=model,
    )
    judge = LLMJudge(j_cfg) if judge_provider != "mock" else MockJudge(j_cfg)
    return judge.diagnose(traj)

