"""Post-hoc confidence calibration models and judge reliability weight estimators."""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from traceroot.diagnosis.judge import JudgeDiagnosis


class TemperatureScaler:
    """Parametric post-hoc calibration via Temperature Scaling: p_cal = sigmoid(logit(p) / T)."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.01, float(temperature))
        self.is_fitted = False

    def fit(self, confidences: List[float], labels: List[bool]) -> "TemperatureScaler":
        """Fit optimal temperature parameter T on validation set using NLL loss."""
        eps = 1e-7
        confs = np.clip(np.array(confidences, dtype=np.float64), eps, 1.0 - eps)
        y = np.array(labels, dtype=np.float64)

        # Convert probabilities to logits
        logits = np.log(confs / (1.0 - confs))

        def nll_objective(T_arr):
            T = T_arr[0]
            scaled_logits = logits / max(1e-4, T)
            p = 1.0 / (1.0 + np.exp(-scaled_logits))
            p = np.clip(p, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            return loss

        res = minimize(nll_objective, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        self.is_fitted = True
        return self

    def calibrate(self, confidence: float) -> float:
        """Calibrate a single confidence score."""
        eps = 1e-7
        c = max(eps, min(1.0 - eps, float(confidence)))
        logit = math.log(c / (1.0 - c))
        scaled_logit = logit / self.temperature
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        return float(max(0.0, min(1.0, calibrated)))

    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """Calibrate a batch of confidence scores."""
        return [self.calibrate(c) for c in confidences]


class PlattScaler:
    """Parametric logistic regression calibrator: p_cal = sigmoid(a * p + b)."""

    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False

    def fit(self, confidences: List[float], labels: List[bool]) -> "PlattScaler":
        confs = np.array(confidences, dtype=np.float64).reshape(-1, 1)
        y = np.array(labels, dtype=int)
        self.model.fit(confs, y)
        self.is_fitted = True
        return self

    def calibrate(self, confidence: float) -> float:
        if not self.is_fitted:
            return confidence
        prob = self.model.predict_proba([[confidence]])[0, 1]
        return float(prob)

    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        if not self.is_fitted:
            return confidences
        confs = np.array(confidences, dtype=np.float64).reshape(-1, 1)
        probs = self.model.predict_proba(confs)[:, 1]
        return [float(p) for p in probs]


class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator."""

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, confidences: List[float], labels: List[bool]) -> "IsotonicCalibrator":
        confs = np.array(confidences, dtype=np.float64)
        y = np.array(labels, dtype=np.float64)
        self.model.fit(confs, y)
        self.is_fitted = True
        return self

    def calibrate(self, confidence: float) -> float:
        if not self.is_fitted:
            return confidence
        return float(self.model.predict([confidence])[0])

    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        if not self.is_fitted:
            return confidences
        confs = np.array(confidences, dtype=np.float64)
        return [float(p) for p in self.model.predict(confs)]


class JudgeReliabilityEstimator:
    """Estimates empirical reliability weights w_j for ensemble judges based on validation performance."""

    @staticmethod
    def estimate_weights(
        judge_predictions: Dict[str, List[bool]],
        ground_truth: List[bool],
        smoothing: float = 0.05,
    ) -> Dict[str, float]:
        """Compute normalized reliability weights w_j based on accuracy and Brier score."""
        y_true = np.array(ground_truth, dtype=bool)
        raw_weights: Dict[str, float] = {}

        for judge_name, preds in judge_predictions.items():
            y_pred = np.array(preds, dtype=bool)
            acc = float(np.mean(y_pred == y_true))
            # Smooth weight to avoid zeroing out any judge completely
            raw_weights[judge_name] = max(0.01, acc + smoothing)

        total_w = sum(raw_weights.values()) or 1.0
        normalized_weights = {j: round(w / total_w, 4) for j, w in raw_weights.items()}
        return normalized_weights
