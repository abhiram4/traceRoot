"""Unit tests for calibration metrics, temperature scaling, and reliability diagrams."""

import numpy as np
import pytest
from traceroot.calibration.calibration import (
    IsotonicCalibrator,
    JudgeReliabilityEstimator,
    PlattScaler,
    TemperatureScaler,
)
from traceroot.calibration.metrics import CalibrationMetrics
from traceroot.calibration.reliability import ReliabilityDiagram


def test_calibration_metrics_computation():
    """Test computing ECE, MCE, and Brier Score on synthetic predictions."""
    confidences = [0.9, 0.8, 0.85, 0.4, 0.3, 0.95, 0.2, 0.6]
    labels = [True, True, False, False, False, True, False, True]

    summary = CalibrationMetrics.compute(confidences, labels, n_bins=5)
    assert summary.total_samples == len(confidences)
    assert 0.0 <= summary.expected_calibration_error <= 1.0
    assert 0.0 <= summary.maximum_calibration_error <= 1.0
    assert 0.0 <= summary.brier_score <= 1.0
    assert len(summary.bins) == 5

    # Check ASCII diagram rendering
    ascii_art = ReliabilityDiagram.render_ascii(summary)
    assert "RELIABILITY DIAGRAM" in ascii_art
    assert "ECE:" in ascii_art

    # Check SVG rendering
    svg_str = ReliabilityDiagram.render_svg(summary)
    assert "<svg" in svg_str
    assert "</svg>" in svg_str


def test_temperature_scaler_calibration():
    """Test fitting and applying temperature scaling to overconfident predictions."""
    # Overconfident model: predicted 0.99 for all, but accuracy was only 50%
    confs = [0.99] * 10 + [0.95] * 10
    labels = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5

    scaler = TemperatureScaler()
    scaler.fit(confs, labels)

    assert scaler.is_fitted
    # Temperature should be > 1.0 to soften overconfident probabilities
    assert scaler.temperature > 1.0

    calibrated = scaler.calibrate(0.95)
    assert calibrated < 0.95  # Should pull confidence down towards 0.5


def test_platt_and_isotonic_scalers():
    """Test Platt scaling and Isotonic regression calibrators."""
    confs = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    labels = [False, False, False, True, True, True]

    platt = PlattScaler().fit(confs, labels)
    assert platt.calibrate(0.1) < platt.calibrate(0.9)

    iso = IsotonicCalibrator().fit(confs, labels)
    assert iso.calibrate(0.1) < iso.calibrate(0.9)


def test_judge_reliability_estimator():
    """Test estimating judge reliability weights w_j from validation performance."""
    ground_truth = [True, False, True, True, False]
    judge_preds = {
        "good_judge": [True, False, True, True, False],     # 100% accurate
        "mediocre_judge": [True, True, True, False, False], # 60% accurate
        "poor_judge": [False, True, False, False, True],    # 0% accurate
    }

    weights = JudgeReliabilityEstimator.estimate_weights(judge_preds, ground_truth)
    assert weights["good_judge"] > weights["mediocre_judge"]
    assert weights["mediocre_judge"] > weights["poor_judge"]
    assert pytest.approx(sum(weights.values()), 0.01) == 1.0
