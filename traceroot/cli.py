"""TraceRoot Command-Line Interface (CLI)."""

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from traceroot.annotation.dataset import DatasetManager
from traceroot.annotation.seed_dataset import get_seed_dataset
from traceroot.calibration.calibration import TemperatureScaler
from traceroot.calibration.metrics import CalibrationMetrics
from traceroot.calibration.reliability import ReliabilityDiagram
from traceroot.core.config import JudgeConfig, load_config
from traceroot.core.types import JudgeType
from traceroot.diagnosis.ensemble import EnsembleAggregator
from traceroot.diagnosis.judge import BaseJudge, HeuristicJudge, LLMJudge, MockJudge
from traceroot.diagnosis.propagation import PropagationGraphBuilder
from traceroot.evaluation.benchmark import BenchmarkRunner
from traceroot.reporting.report import ReportGenerator
from traceroot.trace.normalize import TrajectoryNormalizer
from traceroot.trace.schema import NormalizedTrajectory

app = typer.Typer(
    name="traceroot",
    help="TraceRoot: Failure Localization and Diagnosis for Tool-Using AI Agents",
    add_completion=False,
)
console = Console()


def _resolve_judge(
    judge: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[str] = None,
) -> BaseJudge:
    """Resolve and instantiate the requested judge based on CLI flags or config YAML."""
    if judge:
        j_lower = judge.lower()
        if j_lower == "gemini":
            j_cfg = JudgeConfig(
                name="gemini_judge",
                judge_type=JudgeType.LLM,
                provider="gemini",
                model=model or "gemini-2.0-flash",
                temperature=0.0,
            )
            return LLMJudge(j_cfg)
        elif j_lower == "openai":
            j_cfg = JudgeConfig(
                name="openai_judge",
                judge_type=JudgeType.LLM,
                provider="openai",
                model=model or "gpt-4o",
                temperature=0.0,
            )
            return LLMJudge(j_cfg)
        elif j_lower == "anthropic":
            j_cfg = JudgeConfig(
                name="anthropic_judge",
                judge_type=JudgeType.LLM,
                provider="anthropic",
                model=model or "claude-3-5-sonnet-20241022",
                temperature=0.0,
            )
            return LLMJudge(j_cfg)
        elif j_lower == "heuristic":
            return HeuristicJudge()
        else:
            return MockJudge()

    if config:
        cfg = load_config(config)
        if cfg.ensemble.judges:
            j_cfg = cfg.ensemble.judges[0]
            if j_cfg.provider == "mock":
                return MockJudge(j_cfg)
            elif j_cfg.provider == "heuristic":
                return HeuristicJudge(j_cfg)
            else:
                return LLMJudge(j_cfg)

    return MockJudge()


@app.command()
def diagnose(
    trace_path: str = typer.Argument(..., help="Path to raw or normalized trajectory JSON/log file"),
    format: str = typer.Option("terminal", "--format", "-f", help="Output format: terminal, markdown, html, json"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="File path to save generated report"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    use_ensemble: bool = typer.Option(False, "--ensemble", "-e", help="Use multi-judge ensemble"),
):
    """Diagnose an agent execution trajectory, localize root failure, and classify error taxonomy."""
    cfg = load_config(config)
    normalizer = TrajectoryNormalizer()
    trajectory = normalizer.normalize(trace_path)

    # Instantiate Judge
    if use_ensemble and len(cfg.ensemble.judges) > 1:
        judges = [MockJudge(j_cfg) if j_cfg.provider == "mock" else LLMJudge(j_cfg) for j_cfg in cfg.ensemble.judges]
        diagnoses = [j.diagnose(trajectory) for j in judges]
        aggregator = EnsembleAggregator(cfg.ensemble)
        diagnosis = aggregator.aggregate(diagnoses)
    else:
        primary_cfg = cfg.ensemble.judges[0] if cfg.ensemble.judges else None
        judge = MockJudge(primary_cfg) if (not primary_cfg or primary_cfg.provider == "mock") else LLMJudge(primary_cfg)
        diagnosis = judge.diagnose(trajectory)

    pg = PropagationGraphBuilder.build(trajectory, diagnosis)
    report_gen = ReportGenerator()

    if format == "terminal":
        report_gen.display_terminal(trajectory, diagnosis)
    elif format == "markdown":
        md = report_gen.generate_markdown(trajectory, diagnosis, pg)
        if output:
            report_gen.save_report(md, output)
            console.print(f"[green]Saved Markdown report to:[/green] {output}")
        else:
            console.print(md)
    elif format == "html":
        html = report_gen.generate_html(trajectory, diagnosis, pg)
        if output:
            report_gen.save_report(html, output)
            console.print(f"[green]Saved HTML report to:[/green] {output}")
        else:
            console.print(html)
    elif format == "json":
        js = report_gen.generate_json(trajectory, diagnosis, pg)
        if output:
            report_gen.save_report(js, output)
            console.print(f"[green]Saved JSON report to:[/green] {output}")
        else:
            console.print(js)


@app.command()
def benchmark(
    seed: bool = typer.Option(True, "--seed-dataset", "-s", help="Run benchmark on curated seed dataset"),
    extended: bool = typer.Option(False, "--extended", "-e", help="Run on extended 50-trajectory real-world benchmark suite"),
    judge: Optional[str] = typer.Option(None, "--judge", "-j", help="Judge provider: gemini, openai, anthropic, mock, heuristic"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model name (e.g. gemini-2.0-flash, gemini-2.5-flash)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for benchmark results JSON"),
    pace: float = typer.Option(0.5, "--pace", "-p", help="Delay in seconds between API calls to respect free tier rate limits"),
):
    """Run full scientific evaluation answering RQ1 through RQ5."""
    eval_judge = _resolve_judge(judge=judge, model=model, config=config)
    judge_desc = getattr(eval_judge.config, "model", type(eval_judge).__name__) if hasattr(eval_judge, "config") else type(eval_judge).__name__
    provider_name = getattr(eval_judge.config, "provider", "mock") if hasattr(eval_judge, "config") else "mock"

    console.print(f"[bold cyan]Running TraceRoot Benchmark Evaluation...[/bold cyan]")
    console.print(f"• Judge Provider: [bold yellow]{provider_name}[/bold yellow] | Model: [bold green]{judge_desc}[/bold green]")

    if extended:
        from traceroot.annotation.benchmark_suite_50 import get_extended_benchmark_50
        trajs, dataset = get_extended_benchmark_50()
        console.print(f"• Benchmark Suite: [bold cyan]Extended 50 Real-World Trajectories[/bold cyan]")
    else:
        trajs, dataset = get_seed_dataset()
        console.print(f"• Benchmark Suite: [bold cyan]25 Curated Seed Trajectories[/bold cyan]")

    is_live = provider_name not in ("mock", "heuristic")
    delay = pace if is_live else 0.0
    runner = BenchmarkRunner(judge=eval_judge, delay_between_calls=delay)

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Evaluating {len(trajs)} trajectories...", total=len(trajs))

        def on_step(curr, total, traj, diag):
            progress.update(task, advance=1, description=f"Evaluating [{curr}/{total}] {traj.trajectory_id}")

        results = runner.run(trajs, dataset, progress_callback=on_step)

    console.print("\n[bold green]=== BENCHMARK RESULTS SUMMARY ===[/bold green]")
    console.print(f"Total Evaluated Trajectories: {results.total_trajectories}")
    console.print(f"Taxonomy Classification Accuracy: [bold]{results.classification.accuracy:.1%}[/bold] (Macro F1: {results.classification.macro_f1:.3f})")
    console.print(f"Exact Root-Step Localization: [bold]{results.localization.exact_step_accuracy:.1%}[/bold]")
    console.print(f"Tolerance (k=1) Localization: [bold]{results.localization.tolerance_1_accuracy:.1%}[/bold]")
    console.print(f"Human-Judge Agreement (Kappa): [bold]{results.agreement.taxonomy_category_kappa:.3f}[/bold]")
    console.print(f"Uncalibrated ECE: {results.calibration_uncalibrated.expected_calibration_error:.4f} -> Calibrated ECE: [bold]{results.calibration_calibrated.expected_calibration_error:.4f}[/bold]")

    console.print("\n[bold magenta]=== RESEARCH QUESTION FINDINGS ===[/bold magenta]")
    for rq_key, ans in results.rq_answers.model_dump().items():
        console.print(f"• [bold]{rq_key.upper()}:[/bold] {ans['conclusion']}")

    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(json.dumps(results.model_dump(), indent=2, default=str))
        console.print(f"\n[green]Saved full results to:[/green] {output}")


@app.command()
def calibrate(
    val_data: Optional[str] = typer.Option(None, "--val-data", "-v", help="Path to validation annotations JSON"),
):
    """Fit confidence calibration model and display reliability diagram."""
    trajs, dataset = get_seed_dataset()
    judge = MockJudge()
    diagnoses = [judge.diagnose(t) for t in trajs]

    confidences = [d.confidence for d in diagnoses]
    labels = [
        (a.error_category == d.root_cause_category and a.root_cause_step == d.root_cause_step_id)
        for a, d in zip(dataset.records, diagnoses)
    ]

    summary_raw = CalibrationMetrics.compute(confidences, labels, n_bins=5)
    console.print("[bold yellow]Uncalibrated Confidence Reliability:[/bold yellow]")
    console.print(ReliabilityDiagram.render_ascii(summary_raw))

    scaler = TemperatureScaler().fit(confidences, labels)
    calibrated_confs = scaler.calibrate_batch(confidences)
    summary_cal = CalibrationMetrics.compute(calibrated_confs, labels, n_bins=5)

    console.print(f"\n[bold green]Temperature Scaling Applied (T = {scaler.temperature:.2f}):[/bold green]")
    console.print(ReliabilityDiagram.render_ascii(summary_cal))


@app.command()
def export_seed(
    output_dir: str = typer.Option("./data/seed_benchmark", "--output-dir", "-o", help="Target directory for export"),
):
    """Export the 25+ curated benchmark trajectories and gold human annotations to JSON files."""
    trajs, dataset = get_seed_dataset()
    out_path = Path(output_dir)
    trajs_dir = out_path / "trajectories"
    trajs_dir.mkdir(parents=True, exist_ok=True)

    for t in trajs:
        t.save_json(trajs_dir / f"{t.trajectory_id}.json")

    DatasetManager.save_dataset(dataset, out_path / "gold_annotations.json")
    console.print(f"[green]Successfully exported {len(trajs)} trajectories and annotations to:[/green] {out_path}")


@app.command()
def ui(
    trace_path: Optional[str] = typer.Argument(None, help="Path to trajectory JSON/log file or report HTML (defaults to sample trajectory)"),
    judge: Optional[str] = typer.Option("gemini", "--judge", "-j", help="Judge provider: gemini, openai, mock"),
    model: Optional[str] = typer.Option("gemini-2.0-flash", "--model", "-m", help="Judge model name"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the report automatically in the default browser"),
):
    """Launch TraceRoot interactive HTML dashboard for any trajectory."""
    import webbrowser

    if trace_path and str(trace_path).endswith((".html", ".htm")):
        target_html = Path(trace_path)
    else:
        if not trace_path:
            trajs, _ = get_seed_dataset()
            traj = trajs[1]  # seed_02_invalid_plan
        else:
            normalizer = TrajectoryNormalizer()
            traj = normalizer.normalize(trace_path)

        eval_judge = _resolve_judge(judge=judge, model=model)
        diagnosis = eval_judge.diagnose(traj)
        pg = PropagationGraphBuilder.build(traj, diagnosis)
        report_gen = ReportGenerator()
        html_content = report_gen.generate_html(traj, diagnosis, pg)

        out_dir = Path("./reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        target_html = out_dir / f"{traj.trajectory_id}_report.html"
        report_gen.save_report(html_content, str(target_html))

    abs_path = target_html.resolve()
    console.print(f"[bold green]✓ TraceRoot Interactive UI generated:[/bold green] [cyan]{abs_path}[/cyan]")
    if open_browser:
        webbrowser.open(f"file:///{abs_path.as_posix()}")


@app.command()
def listen(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind OTLP server"),
    port: int = typer.Option(4318, "--port", "-p", help="Port to listen for OpenTelemetry traces"),
):
    """Start local OpenTelemetry OTLP receiver (http://localhost:4318/v1/traces)."""
    from traceroot.execution.otel_server import OTelServer
    server = OTelServer(host=host, port=port)
    server.start(blocking=True)


if __name__ == "__main__":
    app()


