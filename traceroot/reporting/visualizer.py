"""Terminal visualizations and Rich tables for TraceRoot failure reports."""

from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from traceroot.core.types import ErrorCategory, PropagationRelation
from traceroot.diagnosis.judge import JudgeDiagnosis
from traceroot.trace.schema import NormalizedTrajectory


class TerminalVisualizer:
    """Renders formatted colorized terminal reports using Rich."""

    @staticmethod
    def display_report(trajectory: NormalizedTrajectory, diagnosis: JudgeDiagnosis) -> None:
        console = Console()

        # Header panel
        status_color = "green" if trajectory.task_success else "red"
        status_text = "SUCCESS" if trajectory.task_success else "FAILED"
        diag_color = "red" if diagnosis.failure_detected else "green"

        title = f"[bold]TraceRoot Failure Attribution Report[/bold] — [bold {status_color}]{status_text}[/bold {status_color}]"
        
        info_lines = [
            f"[bold cyan]Trajectory ID:[/bold cyan] {trajectory.trajectory_id}",
            f"[bold cyan]Task Prompt:[/bold cyan] {trajectory.task_prompt[:120]}...",
            f"[bold cyan]Diagnosis:[/bold cyan] [{diag_color}]{'FAILURE DETECTED' if diagnosis.failure_detected else 'NO FAILURE DETECTED'}[/{diag_color}] (Confidence: {diagnosis.confidence:.1%})",
        ]

        if diagnosis.failure_detected and diagnosis.root_cause_step_id:
            info_lines.append(
                f"[bold red]Root Cause:[/bold red] Step {diagnosis.root_cause_step_id} ([bold yellow]{diagnosis.root_cause_category.value}[/bold yellow]) in phase '{diagnosis.root_cause_phase.value if diagnosis.root_cause_phase else ''}'"
            )
            info_lines.append(f"[bold cyan]Evidence:[/bold cyan] {diagnosis.summary_rationale}")

        console.print(Panel("\n".join(info_lines), title=title, border_style="cyan"))

        # Step propagation table
        table = Table(title="Trajectory Steps & Propagation Cascade", border_style="blue", show_header=True, header_style="bold magenta")
        table.add_column("Step", style="dim", width=8)
        table.add_column("Phase", width=22)
        table.add_column("Outcome", width=12)
        table.add_column("Role / Relationship", width=24)
        table.add_column("Summary & Evidence")

        assessment_map = {sa.step_id: sa for sa in diagnosis.step_assessments}

        for step in trajectory.steps:
            sa = assessment_map.get(step.step_id)
            is_err = sa.is_error if sa else step.is_tool_error
            outcome = "[red]Incorrect[/red]" if is_err else "[green]Valid[/green]"
            
            rel_str = "[green]Normal[/green]"
            if sa and sa.relation == PropagationRelation.ROOT_CAUSE:
                rel_str = "[bold red]ROOT CAUSE[/bold red]"
            elif sa and sa.relation == PropagationRelation.PROPAGATED_CONSEQUENCE:
                rel_str = "[yellow]Propagated Consequence[/yellow]"

            ev = sa.evidence if sa and sa.evidence else step.to_readable_summary()
            table.add_row(
                f"Step {step.step_id}",
                step.phase.value,
                outcome,
                rel_str,
                ev[:80] + ("..." if len(ev) > 80 else ""),
            )

        console.print(table)

        # Print automated remediation patch if failure diagnosed
        if diagnosis.failure_detected and diagnosis.root_cause_step_id:
            from traceroot.diagnosis.remediation import RemediationGenerator
            from rich.syntax import Syntax

            patch = RemediationGenerator.generate(trajectory, diagnosis)
            if patch and (patch.prompt_diff or patch.schema_code_patch):
                console.print("\n[bold cyan]🛠️  Automated Self-Healing Code Patches:[/bold cyan]")
                if patch.prompt_diff:
                    console.print(f"[bold yellow]1. System Prompt Patch:[/bold yellow] [dim]({patch.prompt_guidance})[/dim]")
                    syntax_diff = Syntax(patch.prompt_diff, "diff", theme="monokai", line_numbers=False)
                    console.print(syntax_diff)

                if patch.schema_code_patch:
                    console.print("\n[bold green]2. Pydantic Tool Parameter Validator:[/bold green]")
                    syntax_py = Syntax(patch.schema_code_patch, "python", theme="monokai", line_numbers=False)
                    console.print(syntax_py)

