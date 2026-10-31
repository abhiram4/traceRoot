"""Human annotation schema, multi-annotator records, and dataset models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from traceroot.core.types import ErrorCategory, ExecutionPhase


class HumanAnnotation(BaseModel):
    """Gold-standard failure attribution annotation provided by a human expert."""
    trajectory_id: str
    annotator_id: str = "expert_annotator_1"
    task_success: bool = True
    first_error_step: Optional[int] = None
    root_cause_step: Optional[int] = None
    root_cause_phase: Optional[ExecutionPhase] = None
    error_category: ErrorCategory = ErrorCategory.NO_ERROR
    propagated_errors: List[int] = Field(default_factory=list)
    evidence_rationale: str = ""
    annotator_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiAnnotatorRecord(BaseModel):
    """Multiple human annotations for a single trajectory to evaluate inter-annotator agreement."""
    trajectory_id: str
    annotations: List[HumanAnnotation] = Field(default_factory=list)
    consensus_root_step: Optional[int] = None
    consensus_category: Optional[ErrorCategory] = None
    consensus_phase: Optional[ExecutionPhase] = None
    is_disputed: bool = False


class HumanDataset(BaseModel):
    """Collection of gold human annotations mapped by trajectory ID."""
    name: str = "traceroot_gold_dataset"
    version: str = "1.0.0"
    records: List[HumanAnnotation] = Field(default_factory=list)
    multi_annotator_records: List[MultiAnnotatorRecord] = Field(default_factory=list)

    def get_annotation(self, trajectory_id: str) -> Optional[HumanAnnotation]:
        for r in self.records:
            if r.trajectory_id == trajectory_id:
                return r
        return None
