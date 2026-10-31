"""Phase-to-category mappings, constraint validation, and disambiguation rules."""

from typing import Dict, List, Optional, Set
from traceroot.core.types import ErrorCategory, ExecutionPhase, FailureFamily
from traceroot.taxonomy.categories import TAXONOMY_REGISTRY, TaxonomyDefinition, get_category_definition


class TaxonomyMapper:
    """Manages structural constraints and relationships between execution phases and error categories."""

    def __init__(self):
        self._registry = TAXONOMY_REGISTRY

    def get_allowed_categories_for_phase(self, phase: ExecutionPhase) -> List[ErrorCategory]:
        """Return all error categories legally permitted for a given execution phase."""
        allowed: List[ErrorCategory] = [ErrorCategory.NO_ERROR]
        for cat, defn in self._registry.items():
            if phase in defn.allowed_phases and cat != ErrorCategory.NO_ERROR:
                allowed.append(cat)
        return allowed

    def is_category_valid_for_phase(self, category: ErrorCategory, phase: ExecutionPhase) -> bool:
        """Check if an error category is allowed to occur at the given execution phase."""
        if category == ErrorCategory.NO_ERROR:
            return True
        defn = self._registry.get(category)
        if not defn:
            return False
        return phase in defn.allowed_phases

    def get_categories_by_family(self, family: FailureFamily) -> List[ErrorCategory]:
        """Return all error categories belonging to a failure family."""
        return [cat for cat, defn in self._registry.items() if defn.family == family]

    def get_phase_family_affinity(self, phase: ExecutionPhase) -> Set[FailureFamily]:
        """Get the primary failure families associated with an execution phase."""
        if phase in {ExecutionPhase.TASK_UNDERSTANDING, ExecutionPhase.PLANNING}:
            return {FailureFamily.COGNITIVE}
        elif phase in {ExecutionPhase.TOOL_SELECTION, ExecutionPhase.TOOL_INVOCATION}:
            return {FailureFamily.TOOL_INTERACTION, FailureFamily.ENVIRONMENTAL}
        elif phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION:
            return {FailureFamily.COGNITIVE, FailureFamily.ENVIRONMENTAL, FailureFamily.RECOVERY}
        elif phase == ExecutionPhase.RECOVERY_ADAPTATION:
            return {FailureFamily.RECOVERY}
        elif phase == ExecutionPhase.FINAL_SYNTHESIS:
            return {FailureFamily.SYNTHESIS, FailureFamily.COGNITIVE}
        return set()

    def filter_valid_categories(
        self,
        predicted_categories: List[ErrorCategory],
        phase: ExecutionPhase,
    ) -> List[ErrorCategory]:
        """Filter a list of predicted categories to only those strictly valid for the step phase."""
        valid = [c for c in predicted_categories if self.is_category_valid_for_phase(c, phase)]
        return valid if valid else [ErrorCategory.NO_ERROR]
