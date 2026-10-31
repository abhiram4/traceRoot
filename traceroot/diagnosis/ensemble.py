"""Multi-judge ensemble aggregator and disagreement analyzer."""

from collections import Counter, defaultdict
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from traceroot.core.config import EnsembleConfig
from traceroot.core.types import (
    AggregationStrategy,
    ErrorCategory,
    ExecutionPhase,
    JudgeType,
    PropagationRelation,
)
from traceroot.diagnosis.judge import (
    BaseJudge,
    JudgeDiagnosis,
    StepAssessment,
)
from traceroot.trace.schema import NormalizedTrajectory


class EnsembleHypothesis(BaseModel):
    """A candidate diagnosis hypothesis with its aggregated probability/weight."""
    root_step_id: Optional[int]
    root_category: ErrorCategory
    root_phase: Optional[ExecutionPhase]
    score: float
    supporting_judges: List[str]


class EnsembleDisagreementReport(BaseModel):
    """Analysis of consensus vs divergence across ensemble judges."""
    consensus_ratio: float  # [0.0, 1.0]
    category_entropy: float  # Shannon entropy over predicted categories
    step_variance: float     # Variance in predicted root step IDs
    has_high_disagreement: bool
    disagreement_reasons: List[str] = Field(default_factory=list)


class AggregatedDiagnosis(JudgeDiagnosis):
    """Diagnosis produced by aggregating multiple independent judges."""
    hypotheses: List[EnsembleHypothesis] = Field(default_factory=list)
    disagreement_report: Optional[EnsembleDisagreementReport] = None
    individual_diagnoses: List[JudgeDiagnosis] = Field(default_factory=list)


class EnsembleAggregator:
    """Combines predictions from multiple judges using weighted voting and consensus optimization."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

    def aggregate(
        self,
        diagnoses: List[JudgeDiagnosis],
        weights: Optional[Dict[str, float]] = None,
    ) -> AggregatedDiagnosis:
        """Aggregate a collection of judge diagnoses into a final consensus diagnosis."""
        if not diagnoses:
            raise ValueError("Cannot aggregate empty list of diagnoses.")

        if len(diagnoses) == 1:
            d = diagnoses[0]
            return AggregatedDiagnosis(
                trajectory_id=d.trajectory_id,
                failure_detected=d.failure_detected,
                root_cause_step_id=d.root_cause_step_id,
                root_cause_phase=d.root_cause_phase,
                root_cause_category=d.root_cause_category,
                confidence=d.confidence,
                summary_rationale=d.summary_rationale,
                step_assessments=d.step_assessments,
                propagated_step_ids=d.propagated_step_ids,
                judge_name="single_judge_ensemble",
                judge_type=JudgeType.ENSEMBLE,
                individual_diagnoses=diagnoses,
                hypotheses=[
                    EnsembleHypothesis(
                        root_step_id=d.root_cause_step_id,
                        root_category=d.root_cause_category,
                        root_phase=d.root_cause_phase,
                        score=d.confidence,
                        supporting_judges=[d.judge_name],
                    )
                ],
            )

        trajectory_id = diagnoses[0].trajectory_id
        effective_weights: Dict[str, float] = {}
        for d in diagnoses:
            w = 1.0
            if weights and d.judge_name in weights:
                w = weights[d.judge_name]
            effective_weights[d.judge_name] = max(0.01, w)

        # 1. Failure Detection Consensus
        fail_votes = sum(
            effective_weights[d.judge_name] * (d.confidence if self.config.aggregation_strategy != AggregationStrategy.MAJORITY_VOTE else 1.0)
            for d in diagnoses if d.failure_detected
        )
        total_fail_weight = sum(
            effective_weights[d.judge_name] * (d.confidence if self.config.aggregation_strategy != AggregationStrategy.MAJORITY_VOTE else 1.0)
            for d in diagnoses
        )
        failure_detected = (fail_votes / total_fail_weight) >= 0.5 if total_fail_weight > 0 else False

        # 2. Joint (Step, Category) Hypothesis Scoring
        hypothesis_scores: Dict[Tuple[Optional[int], ErrorCategory, Optional[ExecutionPhase]], float] = defaultdict(float)
        hypothesis_supporters: Dict[Tuple[Optional[int], ErrorCategory, Optional[ExecutionPhase]], List[str]] = defaultdict(list)

        for d in diagnoses:
            key = (d.root_cause_step_id, d.root_cause_category, d.root_cause_phase)
            weight = effective_weights[d.judge_name]
            
            if self.config.aggregation_strategy == AggregationStrategy.MAJORITY_VOTE:
                score_inc = weight
            elif self.config.aggregation_strategy == AggregationStrategy.CONFIDENCE_WEIGHTED:
                score_inc = weight * d.confidence
            else:  # RELIABILITY_WEIGHTED
                score_inc = weight * d.confidence

            hypothesis_scores[key] += score_inc
            hypothesis_supporters[key].append(d.judge_name)

        # Normalize hypothesis scores
        total_hyp_score = sum(hypothesis_scores.values()) or 1.0
        sorted_hypotheses = sorted(
            [
                EnsembleHypothesis(
                    root_step_id=k[0],
                    root_category=k[1],
                    root_phase=k[2],
                    score=score / total_hyp_score,
                    supporting_judges=hypothesis_supporters[k],
                )
                for k, score in hypothesis_scores.items()
            ],
            key=lambda h: h.score,
            reverse=True,
        )

        top_hyp = sorted_hypotheses[0]
        consensus_ratio = top_hyp.score

        # 3. Disagreement Analysis
        category_counts = Counter(d.root_cause_category.value for d in diagnoses)
        total_d = len(diagnoses)
        cat_probs = [cnt / total_d for cnt in category_counts.values()]
        entropy = -sum(p * math.log2(p) for p in cat_probs if p > 0)

        step_nums = [d.root_cause_step_id for d in diagnoses if d.root_cause_step_id is not None]
        mean_step = sum(step_nums) / len(step_nums) if step_nums else 0.0
        step_var = sum((s - mean_step) ** 2 for s in step_nums) / len(step_nums) if len(step_nums) > 1 else 0.0

        has_high_disagreement = consensus_ratio < self.config.min_agreement_threshold
        disagreement_reasons: List[str] = []
        if has_high_disagreement:
            disagreement_reasons.append(
                f"Consensus ratio ({consensus_ratio:.2f}) below threshold ({self.config.min_agreement_threshold:.2f})."
            )
        if entropy > 1.0:
            disagreement_reasons.append(f"High category entropy ({entropy:.2f}) across judges.")
        if step_var > 2.0:
            disagreement_reasons.append(f"Significant step variance ({step_var:.2f}) among candidate root causes.")

        disagreement_report = EnsembleDisagreementReport(
            consensus_ratio=consensus_ratio,
            category_entropy=entropy,
            step_variance=step_var,
            has_high_disagreement=has_high_disagreement,
            disagreement_reasons=disagreement_reasons,
        )

        # 4. Synthesize Step Assessments
        aggregated_step_assessments = self._aggregate_step_assessments(diagnoses, effective_weights)
        propagated_ids = [
            sa.step_id for sa in aggregated_step_assessments
            if sa.relation == PropagationRelation.PROPAGATED_CONSEQUENCE
        ]

        summary = (
            f"Ensemble consensus: Root cause at Step {top_hyp.root_step_id} ({top_hyp.root_category.value}) "
            f"with {top_hyp.score:.1%} agreement across {len(diagnoses)} judges."
        )

        return AggregatedDiagnosis(
            trajectory_id=trajectory_id,
            failure_detected=failure_detected,
            root_cause_step_id=top_hyp.root_step_id,
            root_cause_phase=top_hyp.root_phase,
            root_cause_category=top_hyp.root_category,
            confidence=top_hyp.score,
            summary_rationale=summary,
            step_assessments=aggregated_step_assessments,
            propagated_step_ids=propagated_ids,
            judge_name="ensemble_aggregator",
            judge_type=JudgeType.ENSEMBLE,
            model_name="ensemble",
            hypotheses=sorted_hypotheses,
            disagreement_report=disagreement_report,
            individual_diagnoses=diagnoses,
        )

    def _aggregate_step_assessments(
        self,
        diagnoses: List[JudgeDiagnosis],
        weights: Dict[str, float],
    ) -> List[StepAssessment]:
        """Aggregate per-step ratings across all judge diagnoses."""
        all_step_ids = set()
        for d in diagnoses:
            for sa in d.step_assessments:
                all_step_ids.add(sa.step_id)

        aggregated: List[StepAssessment] = []
        for s_id in sorted(all_step_ids):
            step_assessments = [
                (d.judge_name, d.get_step_assessment(s_id))
                for d in diagnoses if d.get_step_assessment(s_id) is not None
            ]

            if not step_assessments:
                continue

            phase = step_assessments[0][1].phase
            err_score = sum(
                weights[j_name] * (1.0 if sa.is_error else 0.0)
                for j_name, sa in step_assessments
            )
            tot_w = sum(weights[j_name] for j_name, _ in step_assessments)
            is_err = (err_score / tot_w) >= 0.5

            root_score = sum(
                weights[j_name] * (1.0 if sa.is_root_cause else 0.0)
                for j_name, sa in step_assessments
            )
            is_root = (root_score / tot_w) >= 0.5

            # Most common category for this step
            cat_weights: Dict[ErrorCategory, float] = defaultdict(float)
            for j_name, sa in step_assessments:
                cat_weights[sa.error_category] += weights[j_name]
            top_cat = max(cat_weights.items(), key=lambda x: x[1])[0]

            relation = PropagationRelation.NORMAL
            if is_root:
                relation = PropagationRelation.ROOT_CAUSE
            elif is_err:
                relation = PropagationRelation.PROPAGATED_CONSEQUENCE

            evidences = [sa.evidence for _, sa in step_assessments if sa.evidence]
            combined_evidence = "; ".join(evidences[:2])

            aggregated.append(
                StepAssessment(
                    step_id=s_id,
                    phase=phase,
                    is_error=is_err,
                    error_category=top_cat,
                    is_root_cause=is_root,
                    relation=relation,
                    confidence=min(1.0, (err_score if is_err else (tot_w - err_score)) / tot_w),
                    evidence=combined_evidence,
                )
            )

        return aggregated
