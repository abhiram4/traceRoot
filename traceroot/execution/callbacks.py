"""Native framework callback handlers for LangChain, LangGraph, LlamaIndex, and CrewAI."""

from typing import Any, Callable, Dict, List, Optional, Union
from traceroot.core.types import ExecutionPhase
from traceroot.execution.trace_capture import TrajectoryTracer
from traceroot.trace.schema import NormalizedTrajectory


class TraceRootCallbackHandler:
    """LangChain / LangGraph / LlamaIndex compatible callback handler.
    
    Usage:
        from traceroot.execution.callbacks import TraceRootCallbackHandler
        
        handler = TraceRootCallbackHandler(task_prompt="Book hotel in Paris")
        agent_executor.invoke({"input": "..."}, config={"callbacks": [handler]})
        
        trajectory = handler.get_trajectory()
        diagnosis = handler.diagnose(judge_provider="gemini")
    """

    def __init__(
        self,
        task_prompt: Optional[str] = None,
        task_id: Optional[str] = None,
        on_complete: Optional[Callable[[NormalizedTrajectory], None]] = None,
        auto_diagnose: bool = False,
        judge_provider: str = "gemini",
    ):
        self.task_prompt = task_prompt or "LangChain Agent Execution"
        self.tracer = TrajectoryTracer(task_prompt=self.task_prompt, task_id=task_id, framework="langchain")
        self.on_complete = on_complete
        self.auto_diagnose = auto_diagnose
        self.judge_provider = judge_provider
        self._trajectory: Optional[NormalizedTrajectory] = None

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent chain start."""
        if inputs and isinstance(inputs, dict):
            input_val = inputs.get("input") or inputs.get("question") or inputs.get("prompt")
            if input_val and not self.task_prompt:
                self.task_prompt = str(input_val)
                self.tracer.task_prompt = str(input_val)

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start."""
        pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Handle LLM thought/reasoning output."""
        try:
            if hasattr(response, "generations") and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        text = getattr(gen, "text", "")
                        if text and len(text.strip()) > 0:
                            self.tracer.log_thought(text.strip())
        except Exception:
            pass

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        input_str: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool invocation start."""
        tool_name = "unknown_tool"
        if serialized and isinstance(serialized, dict):
            tool_name = serialized.get("name", "tool")
        elif "name" in kwargs:
            tool_name = kwargs["name"]

        args: Dict[str, Any] = {}
        if isinstance(input_str, dict):
            args = input_str
        elif isinstance(input_str, str):
            args = {"input": input_str}

        self.tracer.log_tool_call(tool_name, args)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Handle tool execution result."""
        tool_name = kwargs.get("name", "tool")
        is_err = "error" in str(output).lower() or "exception" in str(output).lower()
        self.tracer.log_tool_result(
            tool_name=tool_name,
            result=output,
            error=str(output) if is_err else None,
            is_error=is_err,
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Handle tool exception."""
        tool_name = kwargs.get("name", "tool")
        self.tracer.log_tool_result(
            tool_name=tool_name,
            result=None,
            error=str(error),
            is_error=True,
        )

    def on_chain_end(self, outputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Handle chain termination and trajectory finalization."""
        final_text = ""
        if outputs and isinstance(outputs, dict):
            final_text = str(outputs.get("output") or outputs.get("result") or outputs.get("response") or "")

        self.tracer.log_final_response(final_text, success=True)
        self._trajectory = self.tracer.build_trajectory()

        if self.on_complete:
            self.on_complete(self._trajectory)

        if self.auto_diagnose:
            from traceroot.execution.trace_capture import diagnose
            diagnose(self._trajectory, judge_provider=self.judge_provider)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Handle chain failure."""
        self.tracer.log_final_response(f"Execution Error: {error}", success=False)
        self._trajectory = self.tracer.build_trajectory()

        if self.on_complete:
            self.on_complete(self._trajectory)

        if self.auto_diagnose:
            from traceroot.execution.trace_capture import diagnose
            diagnose(self._trajectory, judge_provider=self.judge_provider)

    def get_trajectory(self) -> NormalizedTrajectory:
        """Retrieve the completed NormalizedTrajectory."""
        if not self._trajectory:
            self._trajectory = self.tracer.build_trajectory()
        return self._trajectory

    def diagnose(self, judge_provider: str = "gemini", model: str = "gemini-2.0-flash") -> Any:
        """Run one-line root-cause diagnosis on the captured trajectory."""
        from traceroot.execution.trace_capture import diagnose
        traj = self.get_trajectory()
        return diagnose(traj, judge_provider=judge_provider, model=model)
