"""Calibration and uncertainty evaluation metrics: ECE, MCE, Brier Score, and NLL."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


class CalibrationBin(BaseModel):
    """Statistics for a single confidence interval bin in a reliability diagram."""
    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    accuracy: float
    calibration_gap: float  # |mean_confidence - accuracy|


class CalibrationSummary(BaseModel):
    """Overall calibration metrics summary."""
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float   # MCE
    brier_score: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    negative_log_likelihood: float
    total_samples: int
    bins: List[CalibrationBin] = Field(default_factory=list)


class CalibrationMetrics:
    """Computes calibration quality metrics comparing predicted probabilities to empirical outcomes."""

    @staticmethod
    def compute(
        confidences: List[float],
        ground_truth_correct: List[bool],
        n_bins: int = 10,
        strategy: str = "uniform",  # "uniform" or "quantile"
    ) -> CalibrationSummary:
        """Compute ECE, MCE, Brier Score and bin statistics."""
        if not confidences or not ground_truth_correct:
            raise ValueError("Confidences and ground truth lists cannot be empty.")
        if len(confidences) != len(ground_truth_correct):
            raise ValueError("Confidences and ground truth must have identical lengths.")

        confs = np.clip(np.array(confidences, dtype=np.float64), 0.0, 1.0)
        labels = np.array(ground_truth_correct, dtype=np.float64)
        N = len(confs)

        # 1. Brier Score & Decomposition
        # BS = mean((p - y)^2)
        brier_score = float(np.mean((confs - labels) ** 2))
        base_rate = float(np.mean(labels))
        brier_uncertainty = float(base_rate * (1.0 - base_rate))

        # 2. Binning
        if strategy == "quantile":
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.quantile(confs, quantiles)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        bins_list: List[CalibrationBin] = []
        ece = 0.0
        mce = 0.0
        brier_reliability = 0.0
        brier_resolution = 0.0

        for b_idx in range(len(bin_edges) - 1):
            low = float(bin_edges[b_idx])
            high = float(bin_edges[b_idx + 1])

            # In interval [low, high] (inclusive of upper bound on last bin)
            if b_idx == len(bin_edges) - 2:
                in_bin = (confs >= low) & (confs <= high)
            else:
                in_bin = (confs >= low) & (confs < high)

            count = int(np.sum(in_bin))
            if count > 0:
                mean_conf = float(np.mean(confs[in_bin]))
                acc = float(np.mean(labels[in_bin]))
                gap = float(abs(mean_conf - acc))
                weight = count / N

                ece += weight * gap
                mce = max(mce, gap)

                brier_reliability += weight * (mean_conf - acc) ** 2
                brier_resolution += weight * (acc - base_rate) ** 2

                bins_list.append(
                    CalibrationBin(
                        bin_index=b_idx,
                        lower_bound=round(low, 4),
                        upper_bound=round(high, 4),
                        count=count,
                        mean_confidence=round(mean_conf, 4),
                        accuracy=round(acc, 4),
                        calibration_gap=round(gap, 4),
                    )
                )
            else:
                bins_list.append(
                    CalibrationBin(
                        bin_index=b_idx,
                        lower_bound=round(low, 4),
                        upper_bound=round(high, 4),
                        count=0,
                        mean_confidence=round((low + high) / 2, 4),
                        accuracy=0.0,
                        calibration_gap=0.0,
                    )
                )

        # 3. Negative Log-Likelihood
        eps = 1e-12
        clipped_confs = np.clip(confs, eps, 1.0 - eps)
        nll = float(-np.mean(labels * np.log(clipped_confs) + (1.0 - labels) * np.log(1.0 - clipped_confs)))

        return CalibrationSummary(
            expected_calibration_error=round(float(ece), 4),
            maximum_calibration_error=round(float(mce), 4),
            brier_score=round(float(brier_score), 4),
            brier_reliability=round(float(brier_reliability), 4),
            brier_resolution=round(float(brier_resolution), 4),
            brier_uncertainty=round(float(brier_uncertainty), 4),
            negative_log_likelihood=round(float(nll), 4),
            total_samples=N,
            bins=bins_list,
        )
