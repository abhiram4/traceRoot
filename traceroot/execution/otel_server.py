"""Local OpenTelemetry OTLP background receiver and live trace listener."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional
from rich.console import Console

from traceroot.core.types import ExecutionPhase
from traceroot.execution.trace_capture import diagnose
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryMetadata, TrajectoryStep

logger = logging.getLogger("traceroot.otel_server")
console = Console()


class OTelTraceParser:
    """Parses standard GenAI OpenTelemetry span batches into NormalizedTrajectory."""

    @classmethod
    def parse_otlp_json(cls, payload: Dict[str, Any]) -> List[NormalizedTrajectory]:
        """Convert standard OTLP JSON /v1/traces payload into NormalizedTrajectory instances."""
        trajectories: List[NormalizedTrajectory] = []
        resource_spans = payload.get("resourceSpans", []) or payload.get("resource_spans", [])

        for rs in resource_spans:
            scope_spans = rs.get("scopeSpans", []) or rs.get("scope_spans", [])
            for ss in scope_spans:
                spans = ss.get("spans", [])
                if not spans:
                    continue

                trace_id = spans[0].get("traceId", "otel_trace")
                steps: List[TrajectoryStep] = []
                task_prompt = "OpenTelemetry Agent Task"
                task_success = True
                final_response = None

                step_id = 1
                for span in spans:
                    name = span.get("name", "")
                    attrs_list = span.get("attributes", [])
                    attrs: Dict[str, Any] = {}
                    for a in attrs_list:
                        k = a.get("key")
                        v_obj = a.get("value", {})
                        v = v_obj.get("stringValue") or v_obj.get("intValue") or v_obj.get("boolValue") or v_obj.get("doubleValue") or str(v_obj)
                        if k:
                            attrs[k] = v

                    # Check for prompt / user query
                    prompt_val = attrs.get("gen_ai.prompt") or attrs.get("prompt")
                    if prompt_val and step_id == 1:
                        task_prompt = str(prompt_val)
                        steps.append(
                            TrajectoryStep(
                                step_id=step_id,
                                phase=ExecutionPhase.TASK_UNDERSTANDING,
                                reasoning_context=task_prompt,
                            )
                        )
                        step_id += 1

                    # Check for tool call
                    tool_name = attrs.get("gen_ai.tool.name") or attrs.get("tool.name") or (name if "tool" in name.lower() else None)
                    if tool_name:
                        raw_args = attrs.get("gen_ai.tool.arguments") or attrs.get("tool.arguments") or {}
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except Exception:
                                raw_args = {"_raw": raw_args}

                        steps.append(
                            TrajectoryStep(
                                step_id=step_id,
                                phase=ExecutionPhase.TOOL_INVOCATION,
                                tool_name=str(tool_name),
                                tool_arguments=raw_args if isinstance(raw_args, dict) else {"input": raw_args},
                            )
                        )
                        step_id += 1

                        tool_res = attrs.get("gen_ai.tool.result") or attrs.get("tool.result") or attrs.get("tool.output")
                        if tool_res is not None:
                            is_err = "error" in str(tool_res).lower() or "exception" in str(tool_res).lower()
                            if is_err:
                                task_success = False
                            steps.append(
                                TrajectoryStep(
                                    step_id=step_id,
                                    phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
                                    tool_result=tool_res,
                                    tool_error=str(tool_res) if is_err else None,
                                    is_tool_error=is_err,
                                )
                            )
                            step_id += 1

                    # Check for completion
                    comp_val = attrs.get("gen_ai.completion") or attrs.get("completion")
                    if comp_val:
                        final_response = str(comp_val)

                if final_response:
                    steps.append(
                        TrajectoryStep(
                            step_id=step_id,
                            phase=ExecutionPhase.FINAL_SYNTHESIS,
                            reasoning_context=final_response,
                        )
                    )

                if steps:
                    trajectories.append(
                        NormalizedTrajectory(
                            trajectory_id=str(trace_id),
                            task_prompt=task_prompt,
                            steps=steps,
                            final_response=final_response,
                            task_success=task_success,
                            metadata=TrajectoryMetadata(
                                task_id=str(trace_id),
                                framework="opentelemetry",
                            ),
                        )
                    )

        return trajectories


class OTelHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for receiving OTLP JSON /v1/traces payloads."""

    def do_POST(self) -> None:
        if self.path.startswith("/v1/traces"):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode("utf-8"))
                trajectories = OTelTraceParser.parse_otlp_json(payload)

                for traj in trajectories:
                    console.print(f"[bold cyan]OTel Receiver:[/bold cyan] Ingested trace [yellow]{traj.trajectory_id}[/yellow] ({len(traj.steps)} steps)")
                    # Run live root-cause diagnosis
                    diag = diagnose(traj, judge_provider="gemini")
                    if diag.failure_detected:
                        console.print(f"  [bold red]↳ Root Cause Detected:[/bold red] Step {diag.root_cause_step_id} ({diag.root_cause_category.value})")
                    else:
                        console.print(f"  [bold green]↳ No failure detected.[/bold green]")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "success"}')
            except Exception as e:
                logger.error(f"OTel Receiver error: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default server access logs in terminal


class OTelServer:
    """Standalone background OpenTelemetry receiver."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4318):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = False) -> None:
        """Start the OTLP server."""
        self.server = HTTPServer((self.host, self.port), OTelHTTPHandler)
        console.print(f"[bold green]✓ TraceRoot OTel Receiver listening on:[/bold green] [cyan]http://{self.host}:{self.port}/v1/traces[/cyan]")

        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the OTLP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            console.print("[yellow]TraceRoot OTel Receiver stopped.[/yellow]")
