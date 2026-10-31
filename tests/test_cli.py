"""Unit tests for TraceRoot CLI commands."""

from pathlib import Path
import pytest
from typer.testing import CliRunner
from traceroot.annotation.seed_dataset import get_seed_dataset
from traceroot.cli import app

runner = CliRunner()


def test_cli_diagnose(tmp_path: Path):
    """Test `traceroot diagnose` command on a saved trajectory."""
    trajs, _ = get_seed_dataset()
    sample_traj = trajs[1]  # seed_02_invalid_plan

    traj_file = tmp_path / "sample.json"
    sample_traj.save_json(traj_file)

    # 1. Terminal output
    res = runner.invoke(app, ["diagnose", str(traj_file), "--format", "terminal"])
    assert res.exit_code == 0
    assert "TraceRoot Failure Attribution Report" in res.stdout

    # 2. Markdown output to file
    out_md = tmp_path / "report.md"
    res_md = runner.invoke(app, ["diagnose", str(traj_file), "--format", "markdown", "--output", str(out_md)])
    assert res_md.exit_code == 0
    assert out_md.exists()

    # 3. HTML output to file
    out_html = tmp_path / "report.html"
    res_html = runner.invoke(app, ["diagnose", str(traj_file), "--format", "html", "--output", str(out_html)])
    assert res_html.exit_code == 0
    assert out_html.exists()


def test_cli_benchmark(tmp_path: Path):
    """Test `traceroot benchmark` command."""
    out_res = tmp_path / "benchmark_res.json"
    res = runner.invoke(app, ["benchmark", "--seed-dataset", "--judge", "mock", "--output", str(out_res)])
    assert res.exit_code == 0
    assert "BENCHMARK RESULTS SUMMARY" in res.stdout
    assert "RESEARCH QUESTION FINDINGS" in res.stdout
    assert out_res.exists()


def test_cli_calibrate():
    """Test `traceroot calibrate` command."""
    res = runner.invoke(app, ["calibrate"])
    assert res.exit_code == 0
    assert "RELIABILITY DIAGRAM" in res.stdout
    assert "Temperature Scaling Applied" in res.stdout


def test_cli_export_seed(tmp_path: Path):
    """Test `traceroot export-seed` command."""
    export_dir = tmp_path / "exported_seed"
    res = runner.invoke(app, ["export-seed", "--output-dir", str(export_dir)])
    assert res.exit_code == 0
    assert (export_dir / "gold_annotations.json").exists()
    assert (export_dir / "trajectories").exists()


def test_cli_ui():
    """Test `traceroot ui` command generating interactive dashboard."""
    res = runner.invoke(app, ["ui", "--judge", "mock", "--no-open"])
    assert res.exit_code == 0
    assert "TraceRoot Interactive UI generated" in res.stdout


def test_sdk_trace_decorator():
    """Test high-level @traceroot.trace and diagnose SDK functions."""
    import traceroot

    @traceroot.trace("Search user profile")
    def run_agent(tracer: traceroot.TrajectoryTracer):
        tracer.log_thought("Looking up user")
        tracer.log_tool_call("fetch_user", {"user_id": 42})
        tracer.log_tool_result("fetch_user", {"name": "Alice"})
        tracer.log_final_response("User is Alice", success=True)
        return "Alice"

    result, traj = run_agent()
    assert result == "Alice"
    assert len(traj.steps) == 5
    assert traj.task_success is True

    # Test one-line diagnose
    diagnosis = traceroot.diagnose(traj, judge_provider="mock")
    assert diagnosis is not None
    assert diagnosis.failure_detected is False


def test_cli_benchmark_extended():
    """Test `traceroot benchmark --extended` on 50 real-world trajectories."""
    res = runner.invoke(app, ["benchmark", "--extended", "--judge", "mock"])
    assert res.exit_code == 0
    assert "Extended 50 Real-World Trajectories" in res.stdout
    assert "Total Evaluated Trajectories: 50" in res.stdout



