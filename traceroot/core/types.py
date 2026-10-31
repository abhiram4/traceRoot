"""Core enumeration types and type aliases for TraceRoot."""

from enum import Enum
from typing import Optional


class ExecutionPhase(str, Enum):
    """The 7 standard execution phases of a tool-using agent trajectory."""
    TASK_UNDERSTANDING = "task_understanding"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_INVOCATION = "tool_invocation"
    TOOL_RESULT_INTERPRETATION = "tool_result_interpretation"
    RECOVERY_ADAPTATION = "recovery_adaptation"
    FINAL_SYNTHESIS = "final_synthesis"

    @classmethod
    def from_str(cls, value: str) -> "ExecutionPhase":
        """Convert string to ExecutionPhase safely."""
        clean_val = value.strip().lower().replace(" ", "_").replace("-", "_")
        for phase in cls:
            if phase.value == clean_val:
                return phase
        raise ValueError(f"Unknown execution phase: {value}")


class FailureFamily(str, Enum):
    """High-level family groupings for agent failures."""
    COGNITIVE = "cognitive"
    TOOL_INTERACTION = "tool_interaction"
    ENVIRONMENTAL = "environmental"
    RECOVERY = "recovery"
    SYNTHESIS = "synthesis"
    NONE = "none"


class ErrorCategory(str, Enum):
    """Fine-grained error taxonomy categories for tool-using agents.
    
    Includes the 12-category baseline from 'When Agents Fail to Act' + extensions.
    """
    # Success / No error
    NO_ERROR = "no_error"

    # Cognitive Failures
    MISUNDERSTOOD_TASK = "misunderstood_task"
    INVALID_PLAN = "invalid_plan"
    REASONING_HALLUCINATION = "reasoning_hallucination"

    # Tool Interaction Failures
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    MALFORMED_ARGUMENTS = "malformed_arguments"
    INVALID_API_USAGE = "invalid_api_usage"
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"

    # Environmental Failures
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    ENVIRONMENT_STATE_INCONSISTENCY = "environment_state_inconsistency"
    UNEXPECTED_OBSERVATION = "unexpected_observation"

    # Recovery Failures
    UNHANDLED_TOOL_ERROR = "unhandled_tool_error"
    REPETITIVE_FAILING_ACTION = "repetitive_failing_action"
    PREMATURE_TERMINATION = "premature_termination"

    # Synthesis Failures
    UNSUPPORTED_CONCLUSION = "unsupported_conclusion"
    EVIDENCE_OMISSION = "evidence_omission"
    CONSTRAINT_VIOLATION = "constraint_violation"

    @property
    def family(self) -> FailureFamily:
        """Get the FailureFamily associated with this error category."""
        if self in {
            ErrorCategory.MISUNDERSTOOD_TASK,
            ErrorCategory.INVALID_PLAN,
            ErrorCategory.REASONING_HALLUCINATION,
        }:
            return FailureFamily.COGNITIVE
        elif self in {
            ErrorCategory.WRONG_TOOL_SELECTED,
            ErrorCategory.MALFORMED_ARGUMENTS,
            ErrorCategory.INVALID_API_USAGE,
            ErrorCategory.MISSING_REQUIRED_PARAMETER,
        }:
            return FailureFamily.TOOL_INTERACTION
        elif self in {
            ErrorCategory.TOOL_EXECUTION_ERROR,
            ErrorCategory.ENVIRONMENT_STATE_INCONSISTENCY,
            ErrorCategory.UNEXPECTED_OBSERVATION,
        }:
            return FailureFamily.ENVIRONMENTAL
        elif self in {
            ErrorCategory.UNHANDLED_TOOL_ERROR,
            ErrorCategory.REPETITIVE_FAILING_ACTION,
            ErrorCategory.PREMATURE_TERMINATION,
        }:
            return FailureFamily.RECOVERY
        elif self in {
            ErrorCategory.UNSUPPORTED_CONCLUSION,
            ErrorCategory.EVIDENCE_OMISSION,
            ErrorCategory.CONSTRAINT_VIOLATION,
        }:
            return FailureFamily.SYNTHESIS
        return FailureFamily.NONE


class PropagationRelation(str, Enum):
    """The role a trajectory step plays in failure propagation."""
    NORMAL = "normal"
    ROOT_CAUSE = "root_cause"
    PROPAGATED_CONSEQUENCE = "propagated_consequence"
    INDEPENDENT_FAILURE = "independent_failure"
    RECOVERED_ERROR = "recovered_error"


class PropagationEdgeType(str, Enum):
    """The causal mechanism linking an upstream failure to a downstream step."""
    CAUSAL_DEPENDENCY = "causal_dependency"
    DATA_FLOW = "data_flow"
    MISINTERPRETATION_FLOW = "misinterpretation_flow"
    STATE_POLLUTION = "state_pollution"
    RECOVERY_ATTEMPT = "recovery_attempt"


class ConfidenceLevel(str, Enum):
    """Discrete confidence levels mapped from continuous probabilities."""
    VERY_HIGH = "very_high"  # >= 0.90
    HIGH = "high"            # 0.75 - 0.89
    MEDIUM = "medium"        # 0.50 - 0.74
    LOW = "low"              # < 0.50

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        if score >= 0.90:
            return cls.VERY_HIGH
        elif score >= 0.75:
            return cls.HIGH
        elif score >= 0.50:
            return cls.MEDIUM
        return cls.LOW


class JudgeType(str, Enum):
    """Types of judges available in TraceRoot."""
    LLM = "llm"
    MOCK = "mock"
    HEURISTIC = "heuristic"
    ENSEMBLE = "ensemble"


class AggregationStrategy(str, Enum):
    """Ensemble judge aggregation mechanisms."""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RELIABILITY_WEIGHTED = "reliability_weighted"
