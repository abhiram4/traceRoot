"""Script to compute confidence calibration and export publication SVG diagrams."""

import json
from pathlib import Path
import sys

# Ensure traceroot package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traceroot.annotation.seed_dataset import get_seed_dataset
from traceroot.calibration.calibration import TemperatureScaler
from traceroot.calibration.metrics import CalibrationMetrics
from traceroot.calibration.reliability import ReliabilityDiagram
from traceroot.diagnosis.judge import MockJudge
from traceroot.evaluation.benchmark import BenchmarkRunner


def export_charts():
    reports_dir = Path("./reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run benchmark evaluation on seed dataset
    trajectories, dataset = get_seed_dataset()
    runner = BenchmarkRunner(judge=MockJudge())
    results = runner.run(trajectories, dataset)

    # 2. Extract confidences and accuracies
    confs = [r.confidence for r in results.diagnoses]
    accs = [r.failure_detected for r in results.diagnoses]

    # 3. Compute calibration summary and generate SVG
    cal_summary = CalibrationMetrics.compute(confs, accs, n_bins=5)
    svg_content = ReliabilityDiagram.render_svg(cal_summary)

    svg_path = reports_dir / "reliability_diagram.svg"
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"✓ Exported reliability diagram SVG to: {svg_path}")


if __name__ == "__main__":
    export_charts()
