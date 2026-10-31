"""Standard schema and data models for normalized agent trajectories."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from traceroot.core.types import ExecutionPhase


class ToolCallRecord(BaseModel):
    """Structured representation of a single tool invocation call."""
    call_id: str = Field(default_factory=lambda: f"call_{datetime.now(timezone.utc).timestamp():.6f}")
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    raw_arguments: Optional[str] = None


class ToolResultRecord(BaseModel):
    """Structured representation of an observation/tool execution result."""
    call_id: str
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    is_error: bool = False
    execution_time_ms: Optional[float] = None


class AgentState(BaseModel):
    """Snapshot of agent internal state at a specific step."""
    turn_index: int = 0
    memory_summary: Optional[str] = None
    active_subgoals: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    """An atomic execution step within a multi-step agent trajectory."""
    step_id: int
    phase: ExecutionPhase
    turn_id: int = 1
    implicit: bool = False
    
    # State & Context
    agent_state_before: Optional[str] = None
    reasoning_context: Optional[str] = None
    
    # Tool Interaction (if applicable to phase)
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    raw_tool_arguments: Optional[str] = None
    tool_result: Optional[Any] = None
    tool_error: Optional[str] = None
    is_tool_error: bool = False
    
    # Output State
    agent_state_after: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("phase", mode="before")
    @classmethod
    def validate_phase(cls, v: Any) -> ExecutionPhase:
        if isinstance(v, str):
            return ExecutionPhase.from_str(v)
        return v

    def to_readable_summary(self) -> str:
        """Produce a concise one-line textual description of the step."""
        implicit_tag = " [implicit]" if self.implicit else ""
        if self.phase == ExecutionPhase.TASK_UNDERSTANDING:
            return f"Step {self.step_id} (Task Understanding{implicit_tag}): {self.reasoning_context or 'Initialized task context'}"
        elif self.phase == ExecutionPhase.PLANNING:
            return f"Step {self.step_id} (Planning{implicit_tag}): {self.reasoning_context or 'Formulated execution strategy'}"
        elif self.phase == ExecutionPhase.TOOL_SELECTION:
            return f"Step {self.step_id} (Tool Selection): Selected tool '{self.tool_name}'"
        elif self.phase == ExecutionPhase.TOOL_INVOCATION:
            args_str = json.dumps(self.tool_arguments or {})
            return f"Step {self.step_id} (Tool Invocation): Called '{self.tool_name}' with args {args_str}"
        elif self.phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION:
            status = "ERROR" if self.is_tool_error else "OK"
            res_preview = str(self.tool_result or self.tool_error or "")[:100]
            return f"Step {self.step_id} (Result Interpretation): [{status}] Observed '{res_preview}...'"
        elif self.phase == ExecutionPhase.RECOVERY_ADAPTATION:
            return f"Step {self.step_id} (Recovery): {self.reasoning_context or 'Adapted strategy after failure'}"
        elif self.phase == ExecutionPhase.FINAL_SYNTHESIS:
            return f"Step {self.step_id} (Final Synthesis): Generated final answer"
        return f"Step {self.step_id} ({self.phase.value}): {self.reasoning_context or ''}"


class TrajectoryMetadata(BaseModel):
    """Metadata regarding the benchmark task and agent execution environment."""
    benchmark_name: Optional[str] = "custom"
    task_id: str = "default"
    agent_name: Optional[str] = "unknown_agent"
    model_name: Optional[str] = "unknown_model"
    framework: Optional[str] = "inspect_ai"  # "inspect_ai", "react", "langchain", "native"
    session_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_tokens: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class NormalizedTrajectory(BaseModel):
    """A fully normalized and segmented agent execution trajectory."""
    trajectory_id: str
    task_prompt: str
    task_constraints: List[str] = Field(default_factory=list)
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[TrajectoryStep] = Field(default_factory=list)
    final_response: Optional[str] = None
    task_success: Optional[bool] = None
    ground_truth_answer: Optional[str] = None
    metadata: TrajectoryMetadata = Field(default_factory=TrajectoryMetadata)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def get_step(self, step_id: int) -> Optional[TrajectoryStep]:
        """Retrieve a step by its integer ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_tool_calls(self) -> List[TrajectoryStep]:
        """Retrieve all steps corresponding to tool invocations."""
        return [s for s in self.steps if s.phase == ExecutionPhase.TOOL_INVOCATION]

    def get_steps_by_phase(self, phase: ExecutionPhase) -> List[TrajectoryStep]:
        """Retrieve all steps of a given execution phase."""
        return [s for s in self.steps if s.phase == phase]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trajectory to dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize trajectory to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save normalized trajectory to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizedTrajectory":
        """Construct NormalizedTrajectory from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "NormalizedTrajectory":
        """Construct NormalizedTrajectory from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "NormalizedTrajectory":
        """Load normalized trajectory from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
