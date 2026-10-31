"""Category registry and structured definitions for the TraceRoot error taxonomy."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from traceroot.core.types import ErrorCategory, ExecutionPhase, FailureFamily


class TaxonomyDefinition(BaseModel):
    """Rich definition of an error category with disambiguation examples and interventions."""
    id: ErrorCategory
    name: str
    family: FailureFamily
    description: str
    allowed_phases: List[ExecutionPhase]
    positive_examples: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    recommended_interventions: List[str] = Field(default_factory=list)


# Complete registry of failure taxonomy definitions
TAXONOMY_REGISTRY: Dict[ErrorCategory, TaxonomyDefinition] = {
    ErrorCategory.NO_ERROR: TaxonomyDefinition(
        id=ErrorCategory.NO_ERROR,
        name="No Error (Success / Valid Step)",
        family=FailureFamily.NONE,
        description="The step was executed correctly without errors given available context.",
        allowed_phases=[p for p in ExecutionPhase],
        positive_examples=["Agent correctly searched the order database.", "Agent summarized facts accurately."],
        negative_examples=["Agent skipped required subgoal."],
        recommended_interventions=["No intervention required."],
    ),
    # --- COGNITIVE FAMILY ---
    ErrorCategory.MISUNDERSTOOD_TASK: TaxonomyDefinition(
        id=ErrorCategory.MISUNDERSTOOD_TASK,
        name="Task Understanding Failure",
        family=FailureFamily.COGNITIVE,
        description="Agent misinterprets user instructions, constraints, or expected output goals.",
        allowed_phases=[ExecutionPhase.TASK_UNDERSTANDING, ExecutionPhase.PLANNING],
        positive_examples=[
            "User requested flight for June 15, agent booked for July 15.",
            "User asked to delete temporary files, agent attempted to delete production tables."
        ],
        negative_examples=[
            "Agent understood the task but selected the wrong database query tool (Wrong Tool Selected)."
        ],
        recommended_interventions=[
            "Add explicit prompt constraint reflection / checklist validation.",
            "Implement two-phase task comprehension confirmation step."
        ],
    ),
    ErrorCategory.INVALID_PLAN: TaxonomyDefinition(
        id=ErrorCategory.INVALID_PLAN,
        name="Invalid Plan / Subgoal Decomposition",
        family=FailureFamily.COGNITIVE,
        description="Agent constructs an impossible, circular, unnecessary, or missing-subgoal execution plan.",
        allowed_phases=[ExecutionPhase.PLANNING],
        positive_examples=[
            "Agent planned to write to a file before retrieving the content to write.",
            "Agent omitted the authentication step before attempting private API access."
        ],
        negative_examples=[
            "Agent formulated a valid plan but executed the API call with wrong syntax (Malformed Arguments)."
        ],
        recommended_interventions=[
            "Add topological dependency validation to agent planning module.",
            "Introduce structured subgoal verification before tool dispatch."
        ],
    ),
    ErrorCategory.REASONING_HALLUCINATION: TaxonomyDefinition(
        id=ErrorCategory.REASONING_HALLUCINATION,
        name="Reasoning Hallucination / Deductive Leap",
        family=FailureFamily.COGNITIVE,
        description="Agent invents facts, assumes false preconditions, or makes unjustified logical deductions.",
        allowed_phases=[ExecutionPhase.PLANNING, ExecutionPhase.TOOL_RESULT_INTERPRETATION, ExecutionPhase.FINAL_SYNTHESIS],
        positive_examples=[
            "Tool returned 'User not found', agent assumed 'User account must be suspended for non-payment'.",
            "Agent hallucinated that an email had already been sent without tool evidence."
        ],
        negative_examples=[
            "Tool returned corrupted data that the agent faithfully reported (Unexpected Observation)."
        ],
        recommended_interventions=[
            "Enforce strict evidence citation grounded in tool observations.",
            "Add hallucination guardrail / verification check on reasoning steps."
        ],
    ),
    # --- TOOL INTERACTION FAMILY ---
    ErrorCategory.WRONG_TOOL_SELECTED: TaxonomyDefinition(
        id=ErrorCategory.WRONG_TOOL_SELECTED,
        name="Inappropriate Tool Selection",
        family=FailureFamily.TOOL_INTERACTION,
        description="Agent selects an irrelevant tool when a suitable tool was available, or fails to use tools.",
        allowed_phases=[ExecutionPhase.TOOL_SELECTION],
        positive_examples=[
            "Agent used 'web_search' to find internal company employee records instead of 'internal_hr_directory'.",
            "Agent used 'calculator' tool when asked to translate text."
        ],
        negative_examples=[
            "Agent selected the right tool but passed invalid JSON parameters (Malformed Arguments)."
        ],
        recommended_interventions=[
            "Improve tool description semantic clarity and metadata tags.",
            "Add a tool routing / retrieval step (RAG over tool definitions)."
        ],
    ),
    ErrorCategory.MALFORMED_ARGUMENTS: TaxonomyDefinition(
        id=ErrorCategory.MALFORMED_ARGUMENTS,
        name="Malformed Arguments / Syntax Error",
        family=FailureFamily.TOOL_INTERACTION,
        description="Agent generates unparseable JSON, invalid parameter types, or schema violations.",
        allowed_phases=[ExecutionPhase.TOOL_INVOCATION],
        positive_examples=[
            "Passed string 'true' where integer ID 1 was required.",
            "Generated truncated or unclosed JSON quotes in tool arguments."
        ],
        negative_examples=[
            "Arguments were syntactically valid JSON but contained an invalid username (Invalid API Usage)."
        ],
        recommended_interventions=[
            "Use strict schema enforcement / constrained decoding (e.g., Pydantic / JSON schema mode).",
            "Add parameter type-coercion layer in the tool calling harness."
        ],
    ),
    ErrorCategory.INVALID_API_USAGE: TaxonomyDefinition(
        id=ErrorCategory.INVALID_API_USAGE,
        name="Invalid API Usage / Semantic Argument Error",
        family=FailureFamily.TOOL_INTERACTION,
        description="Syntactically valid arguments that violate semantic API contracts or domain preconditions.",
        allowed_phases=[ExecutionPhase.TOOL_INVOCATION],
        positive_examples=[
            "Passed end_date earlier than start_date to a date-range API.",
            "Queried a SQL table with a column name that does not exist in the database."
        ],
        negative_examples=[
            "Tool threw HTTP 500 server crash due to internal server failure (Tool Execution Error)."
        ],
        recommended_interventions=[
            "Provide few-shot usage examples in tool documentation.",
            "Add pre-execution semantic validator / dry-run linting for tool arguments."
        ],
    ),
    ErrorCategory.MISSING_REQUIRED_PARAMETER: TaxonomyDefinition(
        id=ErrorCategory.MISSING_REQUIRED_PARAMETER,
        name="Missing Required Parameter",
        family=FailureFamily.TOOL_INTERACTION,
        description="Agent omits mandatory parameter fields required for tool execution.",
        allowed_phases=[ExecutionPhase.TOOL_INVOCATION],
        positive_examples=[
            "Called 'send_email' without providing 'recipient_address'.",
            "Called 'search_flights' without 'destination_airport'."
        ],
        negative_examples=[
            "Provided all required parameters, but one parameter had the wrong data type (Malformed Arguments)."
        ],
        recommended_interventions=[
            "Enforce mandatory argument completeness before tool dispatch.",
            "Highlight required fields in tool docstrings."
        ],
    ),
    # --- ENVIRONMENTAL FAMILY ---
    ErrorCategory.TOOL_EXECUTION_ERROR: TaxonomyDefinition(
        id=ErrorCategory.TOOL_EXECUTION_ERROR,
        name="Tool / Environment Runtime Error",
        family=FailureFamily.ENVIRONMENTAL,
        description="Tool crashes, times out, throws HTTP 500, or sandbox environment fails externally.",
        allowed_phases=[ExecutionPhase.TOOL_INVOCATION],
        positive_examples=[
            "Database server returned '503 Service Unavailable'.",
            "Python sandbox container timed out after 30 seconds."
        ],
        negative_examples=[
            "Tool returned '400 Bad Request: invalid query syntax' caused by agent mistake (Invalid API Usage)."
        ],
        recommended_interventions=[
            "Implement exponential backoff and retry policies for external tools.",
            "Add fallback secondary tool endpoints."
        ],
    ),
    ErrorCategory.ENVIRONMENT_STATE_INCONSISTENCY: TaxonomyDefinition(
        id=ErrorCategory.ENVIRONMENT_STATE_INCONSISTENCY,
        name="State Inconsistency / Race Condition",
        family=FailureFamily.ENVIRONMENTAL,
        description="External environment state diverges from expectation (e.g. concurrent mutation).",
        allowed_phases=[ExecutionPhase.TOOL_INVOCATION, ExecutionPhase.TOOL_RESULT_INTERPRETATION],
        positive_examples=[
            "Agent verified item in stock at step 2, but item was sold out when checkout was attempted at step 4.",
            "File deleted by background process during agent execution."
        ],
        negative_examples=[
            "Agent forgot its own previous step results (Reasoning Hallucination)."
        ],
        recommended_interventions=[
            "Add optimistic locking and state re-verification checks.",
            "Implement idempotent multi-step transactions."
        ],
    ),
    ErrorCategory.UNEXPECTED_OBSERVATION: TaxonomyDefinition(
        id=ErrorCategory.UNEXPECTED_OBSERVATION,
        name="Unexpected Observation Format",
        family=FailureFamily.ENVIRONMENTAL,
        description="Tool returns unstructured, corrupted, or schema-divergent content.",
        allowed_phases=[ExecutionPhase.TOOL_RESULT_INTERPRETATION],
        positive_examples=[
            "Tool returned raw HTML error page instead of expected JSON payload.",
            "Tool returned binary garbage bytes."
        ],
        negative_examples=[
            "Tool returned clean JSON error message that the agent ignored (Unhandled Tool Error)."
        ],
        recommended_interventions=[
            "Add tool output sanitization and structured parsing layer.",
            "Normalize tool return payloads into standard schemas before feeding to LLM."
        ],
    ),
    # --- RECOVERY FAMILY ---
    ErrorCategory.UNHANDLED_TOOL_ERROR: TaxonomyDefinition(
        id=ErrorCategory.UNHANDLED_TOOL_ERROR,
        name="Unhandled Tool Error",
        family=FailureFamily.RECOVERY,
        description="Agent receives an explicit tool error but ignores it, continuing as if it succeeded.",
        allowed_phases=[ExecutionPhase.RECOVERY_ADAPTATION, ExecutionPhase.TOOL_RESULT_INTERPRETATION],
        positive_examples=[
            "Tool returned 'Error 404: File not found', but agent stated 'File successfully processed'.",
            "Database insertion failed, but agent reported task completion."
        ],
        negative_examples=[
            "Agent saw the error, tried an alternative query, and succeeded (Valid Recovery)."
        ],
        recommended_interventions=[
            "Add automated reflection prompt whenever a tool returns is_error=True.",
            "Block execution progression until tool errors are acknowledged and resolved."
        ],
    ),
    ErrorCategory.REPETITIVE_FAILING_ACTION: TaxonomyDefinition(
        id=ErrorCategory.REPETITIVE_FAILING_ACTION,
        name="Repetitive Failing Action / Infinite Loop",
        family=FailureFamily.RECOVERY,
        description="Agent retries an identical failing tool call repeatedly without adapting arguments or plan.",
        allowed_phases=[ExecutionPhase.RECOVERY_ADAPTATION, ExecutionPhase.TOOL_INVOCATION],
        positive_examples=[
            "Agent repeated `curl http://broken-link.com` 4 times with identical arguments after receiving 404.",
            "Agent stuck in loop calling search with the exact same misspelled keyword."
        ],
        negative_examples=[
            "Agent modified search query parameters after each failed attempt (Valid Adaptive Retry)."
        ],
        recommended_interventions=[
            "Implement duplicate action detection and loop breaker guardrail.",
            "Inject explicit hint when identical tool call is detected."
        ],
    ),
    ErrorCategory.PREMATURE_TERMINATION: TaxonomyDefinition(
        id=ErrorCategory.PREMATURE_TERMINATION,
        name="Premature Termination",
        family=FailureFamily.RECOVERY,
        description="Agent gives up or concludes execution prematurely when recoverable actions were available.",
        allowed_phases=[ExecutionPhase.RECOVERY_ADAPTATION, ExecutionPhase.FINAL_SYNTHESIS],
        positive_examples=[
            "Agent encountered a single network timeout on tool 1 and immediately responded 'Task impossible' without trying tool 2.",
            "Agent stopped after step 1 of a 4-step workflow."
        ],
        negative_examples=[
            "Agent exhausted all available tools and valid strategies before concluding impossibility (Valid Stop)."
        ],
        recommended_interventions=[
            "Configure agent resilience policy requiring at least N alternative strategies before aborting.",
            "Prompt agent with explicit fallback tools on failure."
        ],
    ),
    # --- SYNTHESIS FAMILY ---
    ErrorCategory.UNSUPPORTED_CONCLUSION: TaxonomyDefinition(
        id=ErrorCategory.UNSUPPORTED_CONCLUSION,
        name="Unsupported Final Conclusion",
        family=FailureFamily.SYNTHESIS,
        description="Final answer makes claims not supported by or directly contradicting collected tool evidence.",
        allowed_phases=[ExecutionPhase.FINAL_SYNTHESIS],
        positive_examples=[
            "Tool observation stated 'Revenue: $5M', final answer reported 'Revenue was $50M'.",
            "Tool returned 0 search results, but agent claimed 'Found 10 matching candidates'."
        ],
        negative_examples=[
            "Agent answered correctly based on wrong tool output caused by earlier bug (Propagated Consequence)."
        ],
        recommended_interventions=[
            "Add synthesis verification checking that all claims cite specific tool observation steps.",
            "Implement NLI (Natural Language Inference) entailment checking between tool observations and final response."
        ],
    ),
    ErrorCategory.EVIDENCE_OMISSION: TaxonomyDefinition(
        id=ErrorCategory.EVIDENCE_OMISSION,
        name="Evidence Omission",
        family=FailureFamily.SYNTHESIS,
        description="Agent retrieved correct evidence during tool steps but failed to include it in final response.",
        allowed_phases=[ExecutionPhase.FINAL_SYNTHESIS],
        positive_examples=[
            "User asked for hotel name and price; agent retrieved both but only listed the hotel name.",
            "Tool found tracking ID FX-99, agent reported 'Package shipped' without the tracking ID."
        ],
        negative_examples=[
            "Agent failed to retrieve the information in the first place (Planning / Tool Selection Failure)."
        ],
        recommended_interventions=[
            "Add response completeness checker against user prompt requirements.",
            "Include structured output schema for final response."
        ],
    ),
    ErrorCategory.CONSTRAINT_VIOLATION: TaxonomyDefinition(
        id=ErrorCategory.CONSTRAINT_VIOLATION,
        name="Output Constraint Violation",
        family=FailureFamily.SYNTHESIS,
        description="Final output violates explicit user format, length, structural, or safety constraints.",
        allowed_phases=[ExecutionPhase.FINAL_SYNTHESIS],
        positive_examples=[
            "User requested output in JSON format only, agent replied with freeform Markdown text.",
            "User requested answer under 50 words, agent produced 500 words."
        ],
        negative_examples=[
            "Agent followed output format but got the wrong answer (Unsupported Conclusion)."
        ],
        recommended_interventions=[
            "Enforce output schema validators on final response.",
            "Add post-processing formatting transformers."
        ],
    ),
}


# Alias for backward compatibility
TaxonomyCategory = TaxonomyDefinition


def get_all_categories() -> List[TaxonomyDefinition]:
    """Retrieve list of all registered taxonomy categories."""
    return list(TAXONOMY_REGISTRY.values())


def get_category_definition(category: ErrorCategory | str) -> Optional[TaxonomyDefinition]:
    """Look up a taxonomy definition by ErrorCategory or string ID."""
    if isinstance(category, str):
        try:
            category = ErrorCategory(category.strip().lower().replace(" ", "_").replace("-", "_"))
        except ValueError:
            return None
    return TAXONOMY_REGISTRY.get(category)
