"""Inter-annotator and LLM-vs-Human agreement metrics: Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha."""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score

from traceroot.annotation.schema import HumanAnnotation, MultiAnnotatorRecord
from traceroot.diagnosis.judge import JudgeDiagnosis


class AgreementReport(BaseModel):
    """Structured report on agreement between human annotators or LLM vs humans."""
    failure_detection_kappa: float
    phase_classification_kappa: float
    taxonomy_category_kappa: float
    taxonomy_category_krippendorff: float
    root_step_krippendorff: float
    percent_agreement_category: float
    percent_agreement_step: float
    total_items: int


class AgreementMetrics:
    """Computes inter-rater reliability metrics."""

    @staticmethod
    def cohens_kappa(rater1: List[Any], rater2: List[Any]) -> float:
        """Calculate Cohen's Kappa coefficient between two raters."""
        if len(rater1) != len(rater2) or not rater1:
            return 0.0
        r1_str = [str(x) for x in rater1]
        r2_str = [str(x) for x in rater2]
        return float(cohen_kappa_score(r1_str, r2_str))

    @staticmethod
    def fleiss_kappa(ratings: List[List[Any]]) -> float:
        """Calculate Fleiss' Kappa for multiple raters on nominal categories.
        
        ratings: List of rows, where each row is [rater1_choice, rater2_choice, ...] for one item.
        """
        if not ratings or not ratings[0]:
            return 0.0

        N = len(ratings)  # number of subjects
        n_raters = len(ratings[0])  # number of ratings per subject
        if n_raters < 2 or N == 0:
            return 1.0

        # Collect all unique categories
        all_cats = sorted(list({str(val) for row in ratings for val in row}))
        k = len(all_cats)
        if k <= 1:
            return 1.0
        cat_to_idx = {c: i for i, c in enumerate(all_cats)}

        # Build table n_ij: count of raters assigning subject i to category j
        table = np.zeros((N, k), dtype=np.float64)
        for i, row in enumerate(ratings):
            for val in row:
                table[i, cat_to_idx[str(val)]] += 1

        # Calculate p_j (proportion of all assignments to category j)
        p_j = np.sum(table, axis=0) / (N * n_raters)

        # Calculate P_i (degree of agreement for subject i)
        P_i = (np.sum(table ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = np.mean(P_i)
        P_e_bar = np.sum(p_j ** 2)

        if abs(1.0 - P_e_bar) < 1e-12:
            return 1.0

        kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
        return float(round(kappa, 4))

    @staticmethod
    def krippendorffs_alpha(
        matrix: List[List[Any]],
        level_of_measurement: str = "nominal",
    ) -> float:
        """Compute Krippendorff's Alpha for nominal or interval data.
        
        matrix: List of items, where each item is a list of ratings from different raters.
        """
        if not matrix:
            return 0.0

        # Flatten valid pairs
        units = matrix
        all_vals = [x for u in units for x in u if x is not None]
        if len(set(all_vals)) <= 1:
            return 1.0

        # For nominal:
        if level_of_measurement == "nominal":
            unique_cats = sorted(list(set(str(v) for v in all_vals)))
            val_map = {c: i for i, c in enumerate(unique_cats)}
            
            # Coincidence matrix
            q = len(unique_cats)
            coincidences = np.zeros((q, q), dtype=np.float64)
            n_total = 0

            for unit in units:
                valid_in_unit = [str(v) for v in unit if v is not None]
                m = len(valid_in_unit)
                if m > 1:
                    for v1 in valid_in_unit:
                        for v2 in valid_in_unit:
                            if v1 != v2 or m > 1:
                                coincidences[val_map[v1], val_map[v2]] += 1.0 / (m - 1)
                                n_total += 1.0 / (m - 1)

            if n_total == 0:
                return 1.0

            n_v = np.sum(coincidences, axis=1)
            # Observed disagreement
            d_o = n_total - np.sum(np.diag(coincidences))
            # Expected disagreement
            d_e = (n_total ** 2 - np.sum(n_v ** 2)) / max(1.0, (n_total - 1)) if n_total > 1 else 1.0

            if d_e == 0:
                return 1.0
            return float(round(1.0 - (d_o / d_e), 4))

        # Interval measurement (e.g. step numbers)
        else:
            diffs_o = []
            all_nums = [float(v) for v in all_vals if v is not None]
            for unit in units:
                nums = [float(v) for v in unit if v is not None]
                if len(nums) > 1:
                    for i in range(len(nums)):
                        for j in range(len(nums)):
                            diffs_o.append((nums[i] - nums[j]) ** 2)

            if not diffs_o:
                return 1.0

            d_o = np.mean(diffs_o)
            # Expected disagreement across all possible pairs
            all_diffs_e = [(a - b) ** 2 for a in all_nums for b in all_nums]
            d_e = np.mean(all_diffs_e) if all_diffs_e else 1.0

            if d_e == 0:
                return 1.0
            return float(round(1.0 - (d_o / d_e), 4))

    @classmethod
    def evaluate_human_llm_agreement(
        cls,
        annotations: List[HumanAnnotation],
        diagnoses: List[JudgeDiagnosis],
    ) -> AgreementReport:
        """Compute full suite of agreement metrics between Human gold labels and LLM diagnoses."""
        diag_map = {d.trajectory_id: d for d in diagnoses}

        h_detect, l_detect = [], []
        h_phase, l_phase = [], []
        h_cat, l_cat = [], []
        h_step, l_step = [], []
        cat_matrix: List[List[Any]] = []
        step_matrix: List[List[Any]] = []

        cat_matches = 0
        step_matches = 0
        total = 0

        for ann in annotations:
            diag = diag_map.get(ann.trajectory_id)
            if not diag:
                continue

            total += 1
            # 1. Detection
            h_detect.append(not ann.task_success)
            l_detect.append(diag.failure_detected)

            # 2. Phase
            h_p = ann.root_cause_phase.value if ann.root_cause_phase else "none"
            l_p = diag.root_cause_phase.value if diag.root_cause_phase else "none"
            h_phase.append(h_p)
            l_phase.append(l_p)

            # 3. Category
            h_c = ann.error_category.value
            l_c = diag.root_cause_category.value
            h_cat.append(h_c)
            l_cat.append(l_c)
            cat_matrix.append([h_c, l_c])
            if h_c == l_c:
                cat_matches += 1

            # 4. Step Number
            h_s = ann.root_cause_step or 0
            l_s = diag.root_cause_step_id or 0
            h_step.append(h_s)
            l_step.append(l_s)
            step_matrix.append([h_s, l_s])
            if h_s == l_s:
                step_matches += 1

        det_k = cls.cohens_kappa(h_detect, l_detect)
        phase_k = cls.cohens_kappa(h_phase, l_phase)
        cat_k = cls.cohens_kappa(h_cat, l_cat)
        cat_alpha = cls.krippendorffs_alpha(cat_matrix, level_of_measurement="nominal")
        step_alpha = cls.krippendorffs_alpha(step_matrix, level_of_measurement="interval")

        return AgreementReport(
            failure_detection_kappa=round(det_k, 4),
            phase_classification_kappa=round(phase_k, 4),
            taxonomy_category_kappa=round(cat_k, 4),
            taxonomy_category_krippendorff=round(cat_alpha, 4),
            root_step_krippendorff=round(step_alpha, 4),
            percent_agreement_category=round(cat_matches / max(1, total), 4),
            percent_agreement_step=round(step_matches / max(1, total), 4),
            total_items=total,
        )
