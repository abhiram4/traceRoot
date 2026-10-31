"""OpenTelemetry GenAI Semantic Conventions and Span Exporter/Importer for TraceRoot.

Follows OpenTelemetry Semantic Conventions for Generative AI Systems:
- gen_ai.system
- gen_ai.agent.name
- gen_ai.operation.name
- gen_ai.tool.name
- gen_ai.tool.call.id
- gen_ai.usage.input_tokens / output_tokens
- error.type
- traceroot.phase
- traceroot.root_cause
- traceroot.error_category
- traceroot.confidence
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from traceroot.core.types import ExecutionPhase
from traceroot.trace.schema import (
    NormalizedTrajectory,
    TrajectoryMetadata,
    TrajectoryStep,
)


class OTelGenAISpan:
    """Represents an OpenTelemetry Span adhering to GenAI semantic conventions."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str] = None,
        start_time_iso: Optional[str] = None,
        end_time_iso: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        status_code: str = "OK",  # "UNSET", "OK", "ERROR"
        status_message: Optional[str] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time_iso = start_time_iso or datetime.now(timezone.utc).isoformat()
        self.end_time_iso = end_time_iso or self.start_time_iso
        self.attributes = attributes or {}
        self.status_code = status_code
        self.status_message = status_message
        self.events = events or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard OpenTelemetry span dictionary."""
        return {
            "name": self.name,
            "context": {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
            },
            "parent_id": self.parent_span_id,
            "start_time": self.start_time_iso,
            "end_time": self.end_time_iso,
            "attributes": self.attributes,
            "status": {
                "code": self.status_code,
                "description": self.status_message or "",
            },
            "events": self.events,
        }


class OpenTelemetryExporter:
    """Exports TraceRoot NormalizedTrajectory instances to OpenTelemetry GenAI spans."""

    @staticmethod
    def trajectory_to_spans(trajectory: NormalizedTrajectory) -> List[Dict[str, Any]]:
        """Convert a normalized trajectory into a list of OpenTelemetry GenAI spans."""
        spans: List[Dict[str, Any]] = []
        trace_id = f"trace_{trajectory.trajectory_id.replace('-', '')[:32].ljust(32, '0')}"
        root_span_id = f"root_{trajectory.trajectory_id[:16]}"

        # 1. Root Agent Run Span
        root_span = OTelGenAISpan(
            name=f"agent.run:{trajectory.metadata.agent_name or 'agent'}",
            trace_id=trace_id,
            span_id=root_span_id,
            attributes={
                "gen_ai.system": trajectory.metadata.framework or "traceroot",
                "gen_ai.agent.name": trajectory.metadata.agent_name or "agent",
                "gen_ai.request.model": trajectory.metadata.model_name or "unknown",
                "gen_ai.operation.name": "agent.execution",
                "traceroot.trajectory_id": trajectory.trajectory_id,
                "traceroot.task_id": trajectory.metadata.task_id,
                "traceroot.task_prompt": trajectory.task_prompt,
                "traceroot.task_success": trajectory.task_success if trajectory.task_success is not None else "unknown",
                "traceroot.total_steps": trajectory.total_steps,
            },
            status_code="ERROR" if trajectory.task_success is False else "OK",
        )
        spans.append(root_span.to_dict())

        # 2. Child Step Spans
        for step in trajectory.steps:
            step_span_id = f"step_{step.step_id}_{root_span_id[:10]}"
            attrs: Dict[str, Any] = {
                "gen_ai.system": trajectory.metadata.framework or "traceroot",
                "gen_ai.operation.name": f"agent.{step.phase.value}",
                "traceroot.step_id": step.step_id,
                "traceroot.phase": step.phase.value,
                "traceroot.implicit": step.implicit,
                "traceroot.turn_id": step.turn_id,
            }

            if step.reasoning_context:
                attrs["gen_ai.choice.reasoning"] = step.reasoning_context

            if step.tool_name:
                attrs["gen_ai.tool.name"] = step.tool_name
            if step.tool_arguments:
                attrs["gen_ai.tool.call.arguments"] = json.dumps(step.tool_arguments)
            if step.tool_result is not None:
                attrs["gen_ai.tool.call.result"] = str(step.tool_result)[:500]

            status = "ERROR" if step.is_tool_error else "OK"
            if step.tool_error:
                attrs["error.type"] = "ToolExecutionError"
                attrs["error.message"] = step.tool_error

            step_span = OTelGenAISpan(
                name=f"agent.step:{step.phase.value}",
                trace_id=trace_id,
                span_id=step_span_id,
                parent_span_id=root_span_id,
                attributes=attrs,
                status_code=status,
                status_message=step.tool_error,
            )
            spans.append(step_span.to_dict())

        return spans

    @staticmethod
    def spans_to_trajectory(spans: List[Dict[str, Any]]) -> NormalizedTrajectory:
        """Parse OpenTelemetry spans back into a NormalizedTrajectory."""
        root_span = next((s for s in spans if not s.get("parent_id")), spans[0] if spans else {})
        root_attrs = root_span.get("attributes", {})

        traj_id = str(root_attrs.get("traceroot.trajectory_id") or root_attrs.get("traceroot.task_id") or "otel_trace_1")
        task_prompt = str(root_attrs.get("traceroot.task_prompt") or "OpenTelemetry Ingested Task")

        steps: List[TrajectoryStep] = []
        for s in spans:
            attrs = s.get("attributes", {})
            if "traceroot.step_id" in attrs or "traceroot.phase" in attrs:
                phase_val = attrs.get("traceroot.phase", ExecutionPhase.PLANNING.value)
                tool_args = None
                if "gen_ai.tool.call.arguments" in attrs:
                    try:
                        tool_args = json.loads(attrs["gen_ai.tool.call.arguments"])
                    except Exception:
                        tool_args = {"_raw": attrs["gen_ai.tool.call.arguments"]}

                step = TrajectoryStep(
                    step_id=int(attrs.get("traceroot.step_id", len(steps) + 1)),
                    phase=ExecutionPhase.from_str(phase_val),
                    turn_id=int(attrs.get("traceroot.turn_id", 1)),
                    implicit=bool(attrs.get("traceroot.implicit", False)),
                    reasoning_context=attrs.get("gen_ai.choice.reasoning"),
                    tool_name=attrs.get("gen_ai.tool.name"),
                    tool_arguments=tool_args,
                    tool_result=attrs.get("gen_ai.tool.call.result"),
                    tool_error=attrs.get("error.message"),
                    is_tool_error=bool(s.get("status", {}).get("code") == "ERROR"),
                )
                steps.append(step)

        # Sort steps by step_id
        steps.sort(key=lambda x: x.step_id)

        return NormalizedTrajectory(
            trajectory_id=traj_id,
            task_prompt=task_prompt,
            steps=steps,
            task_success=root_attrs.get("traceroot.task_success") if isinstance(root_attrs.get("traceroot.task_success"), bool) else None,
            metadata=TrajectoryMetadata(
                task_id=traj_id,
                agent_name=str(root_attrs.get("gen_ai.agent.name") or "otel_agent"),
                model_name=str(root_attrs.get("gen_ai.request.model") or "unknown_model"),
                framework="opentelemetry",
            ),
        )
