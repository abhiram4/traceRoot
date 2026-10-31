"""Unit tests for report generator (Markdown, HTML, JSON, Terminal)."""

from pathlib import Path
import pytest
from traceroot.diagnosis.judge import MockJudge
from traceroot.reporting.report import ReportGenerator
from traceroot.trace.schema import NormalizedTrajectory


def test_report_generator_markdown_html_json(sample_normalized_trajectory: NormalizedTrajectory, tmp_path: Path):
    """Test generating Markdown, HTML, and JSON reports."""
    traj = sample_normalized_trajectory
    judge = MockJudge()
    diagnosis = judge.diagnose(traj)

    generator = ReportGenerator()

    # 1. Markdown
    md_report = generator.generate_markdown(traj, diagnosis)
    assert "## TraceRoot Failure Report" in md_report
    assert traj.trajectory_id in md_report
    assert "### Downstream Propagation" in md_report

    md_file = tmp_path / "report.md"
    generator.save_report(md_report, md_file)
    assert md_file.exists()

    # 2. HTML
    html_report = generator.generate_html(traj, diagnosis)
    assert "<!DOCTYPE html>" in html_report
    assert "TraceRoot Failure Report" in html_report
    assert "mermaid" in html_report

    html_file = tmp_path / "report.html"
    generator.save_report(html_report, html_file)
    assert html_file.exists()

    # 3. JSON
    json_report = generator.generate_json(traj, diagnosis)
    assert "trajectory_id" in json_report
    assert "propagation_graph" in json_report
