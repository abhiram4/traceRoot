"""Counterfactual verification engine for double-checking causal root-cause attributions."""

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


COUNTERFACTUAL_SYSTEM_PROMPT = """You are the TraceRoot Counterfactual Causal Verifier.
Your task is to double-check a candidate root-cause hypothesis using counterfactual necessity and precedence tests.

STRICT CAUSAL VERIFICATION RULES:
1. Necessity Test: If the candidate step had executed validly, would subsequent failures have been avoided? (If YES -> Valid candidate).
2. Precedence Test: Did any EARLIER step before this candidate step already contain an unforced mistake that caused this candidate step? (If YES -> The earlier step is the TRUE root cause).
3. Symptom vs Cause: If the candidate step is merely a tool server error or observation resulting from bad arguments passed earlier, the argument formulation step is the TRUE root cause.

Output valid JSON only.
"""

COUNTERFACTUAL_USER_PROMPT = """Please verify the following causal diagnosis hypothesis:

=== TASK ===
Task Prompt: {{ task_prompt }}

=== CANDIDATE HYPOTHESIS ===
Candidate Root Cause Step: Step {{ candidate_step_id }}
Candidate Error Category: {{ candidate_category }}
Candidate Summary: {{ candidate_summary }}

=== TRAJECTORY STEPS IN CONTEXT ===
{% for step in steps %}
--- Step {{ step.step_id }} [Phase: {{ step.phase.value }}] ---
{% if step.reasoning_context %}Reasoning: {{ step.reasoning_context }}{% endif %}
{% if step.tool_name %}Tool Call: {{ step.tool_name }}({{ step.tool_arguments }}){% endif %}
{% if step.tool_result is not none %}Tool Result: {{ step.tool_result }}{% endif %}
{% if step.tool_error %}Tool Error: {{ step.tool_error }}{% endif %}
{% endfor %}

=== REQUIRED JSON OUTPUT ===
{
  "is_confirmed": true,
  "confidence": 0.95,
  "precedence_test_passed": true,
  "necessity_test_passed": true,
  "true_root_cause_step": {{ candidate_step_id }},
  "true_root_cause_category": "{{ candidate_category }}",
  "verification_rationale": "Explanation of why this candidate satisfies necessity and precedence."
}
"""


class CounterfactualVerificationResult(BaseModel):
    """Result of the counterfactual verification check."""
    is_confirmed: bool
    confidence: float
    true_root_cause_step: Optional[int] = None
    true_root_cause_category: Optional[ErrorCategory] = None
    verification_rationale: str = ""
    necessity_test_passed: bool = True
    precedence_test_passed: bool = True


class CounterfactualVerifier:
    """Verifies candidate root causes by testing counterfactual necessity and temporal precedence."""

    @classmethod
    def verify_offline(
        cls,
        trajectory: NormalizedTrajectory,
        candidate_step_id: int,
        candidate_category: ErrorCategory,
    ) -> CounterfactualVerificationResult:
        """Deterministic offline rule-based verification for testing and fallback."""
        steps_map = {s.step_id: s for s in trajectory.steps}
        candidate_step = steps_map.get(candidate_step_id)

        if not candidate_step:
            return CounterfactualVerificationResult(
                is_confirmed=False,
                confidence=0.5,
                true_root_cause_step=candidate_step_id,
                true_root_cause_category=candidate_category,
                verification_rationale="Candidate step ID not found in trajectory.",
            )

        # Precedence check: Is there an earlier step with explicit error?
        for s in trajectory.steps:
            if s.step_id < candidate_step_id:
                if s.phase == ExecutionPhase.PLANNING and s.reasoning_context and "invalid" in s.reasoning_context.lower():
                    return CounterfactualVerificationResult(
                        is_confirmed=False,
                        confidence=0.92,
                        true_root_cause_step=s.step_id,
                        true_root_cause_category=ErrorCategory.INVALID_PLAN,
                        precedence_test_passed=False,
                        verification_rationale=f"Precedence test failed: Step {s.step_id} (planning) contained an unforced planning error that preceded Step {candidate_step_id}.",
                    )
                if s.phase == ExecutionPhase.TOOL_INVOCATION and s.tool_arguments and "_raw" in s.tool_arguments:
                    return CounterfactualVerificationResult(
                        is_confirmed=False,
                        confidence=0.95,
                        true_root_cause_step=s.step_id,
                        true_root_cause_category=ErrorCategory.MALFORMED_ARGUMENTS,
                        precedence_test_passed=False,
                        verification_rationale=f"Precedence test failed: Step {s.step_id} had malformed tool arguments preceding Step {candidate_step_id}.",
                    )

        return CounterfactualVerificationResult(
            is_confirmed=True,
            confidence=0.94,
            true_root_cause_step=candidate_step_id,
            true_root_cause_category=candidate_category,
            precedence_test_passed=True,
            necessity_test_passed=True,
            verification_rationale=f"Candidate Step {candidate_step_id} ({candidate_category.value}) passed counterfactual necessity and precedence tests.",
        )
