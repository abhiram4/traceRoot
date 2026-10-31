"""Unit tests for error taxonomy definitions and phase mappings."""

import pytest
from traceroot.core.types import ErrorCategory, ExecutionPhase, FailureFamily
from traceroot.taxonomy.categories import (
    get_all_categories,
    get_category_definition,
    TAXONOMY_REGISTRY,
)
from traceroot.taxonomy.mappings import TaxonomyMapper


def test_taxonomy_registry_completeness():
    """Verify all 12 baseline categories and extensions have valid definitions."""
    all_cats = get_all_categories()
    assert len(all_cats) >= 13  # 12 baseline + no_error + extensions

    for cat_defn in all_cats:
        assert cat_defn.name
        assert cat_defn.description
        assert len(cat_defn.allowed_phases) > 0
        assert cat_defn.family in [f for f in FailureFamily]


def test_category_lookup():
    """Test looking up taxonomy definition by enum and string."""
    defn = get_category_definition(ErrorCategory.INVALID_PLAN)
    assert defn is not None
    assert defn.family == FailureFamily.COGNITIVE
    assert ExecutionPhase.PLANNING in defn.allowed_phases

    str_defn = get_category_definition("wrong_tool_selected")
    assert str_defn is not None
    assert str_defn.id == ErrorCategory.WRONG_TOOL_SELECTED

    invalid_lookup = get_category_definition("non_existent_category")
    assert invalid_lookup is None


def test_taxonomy_mapper_phase_constraints():
    """Test phase-to-category constraint rules."""
    mapper = TaxonomyMapper()
    
    # invalid_plan should only be allowed in PLANNING
    assert mapper.is_category_valid_for_phase(ErrorCategory.INVALID_PLAN, ExecutionPhase.PLANNING)
    assert not mapper.is_category_valid_for_phase(ErrorCategory.INVALID_PLAN, ExecutionPhase.TOOL_INVOCATION)

    # malformed_arguments should be allowed in TOOL_INVOCATION
    assert mapper.is_category_valid_for_phase(ErrorCategory.MALFORMED_ARGUMENTS, ExecutionPhase.TOOL_INVOCATION)
    assert not mapper.is_category_valid_for_phase(ErrorCategory.MALFORMED_ARGUMENTS, ExecutionPhase.FINAL_SYNTHESIS)

    # NO_ERROR is universally valid
    for phase in ExecutionPhase:
        assert mapper.is_category_valid_for_phase(ErrorCategory.NO_ERROR, phase)

    # Filter invalid categories
    filtered = mapper.filter_valid_categories(
        [ErrorCategory.INVALID_PLAN, ErrorCategory.MALFORMED_ARGUMENTS],
        ExecutionPhase.TOOL_INVOCATION,
    )
    assert filtered == [ErrorCategory.MALFORMED_ARGUMENTS]
