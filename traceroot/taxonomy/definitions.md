# TraceRoot Error Taxonomy Specification & Disambiguation Guide

This document defines the 12-category baseline taxonomy (extended from *"When Agents Fail to Act"*) used by TraceRoot to classify and attribute failures in tool-using AI agent trajectories.

---

## 1. Cognitive Failure Family (`cognitive`)

### 1.1 `misunderstood_task` (Task Understanding Failure)
- **Phase:** `task_understanding`, `planning`
- **Definition:** The agent misinterprets the user's intent, goals, constraints, or specified preconditions.
- **Example:** User requested data for "March 2024", but the agent formulated an execution strategy for "March 2023".
- **Disambiguation:** If the agent understood the task correctly but chose an incorrect sub-tool, classify as `wrong_tool_selected`.

### 1.2 `invalid_plan` (Invalid Plan / Subgoal Decomposition)
- **Phase:** `planning`
- **Definition:** The agent forms an impossible, circular, unnecessary, or missing-subgoal execution plan.
- **Example:** The agent plans to update database record #42 without first fetching record #42 to obtain the current lock version.
- **Disambiguation:** If the plan was sound but the tool call syntax failed, classify as `malformed_arguments`.

### 1.3 `reasoning_hallucination` (Reasoning Hallucination / Inference Error)
- **Phase:** `planning`, `tool_result_interpretation`, `final_synthesis`
- **Definition:** The agent asserts assumptions or logical deductions that are contradicted or ungrounded by the environment/tools.
- **Example:** Tool returned `user_found: false`; agent stated `User account was deleted due to non-payment`.

---

## 2. Tool Interaction Failure Family (`tool_interaction`)

### 2.1 `wrong_tool_selected` (Inappropriate Tool Selection)
- **Phase:** `tool_selection`
- **Definition:** The agent selects an irrelevant or ineffective tool when an appropriate tool was available.
- **Example:** Agent invoked `calculator` when asked to summarize a PDF document.

### 2.2 `malformed_arguments` (Malformed Arguments / Syntax Error)
- **Phase:** `tool_invocation`
- **Definition:** The agent outputs unparseable JSON, illegal syntax, or wrong data types.
- **Example:** Sent string `"twenty"` where integer `20` was strictly required by tool schema.

### 2.3 `invalid_api_usage` (Invalid API Usage / Semantic Argument Error)
- **Phase:** `tool_invocation`
- **Definition:** Syntactically valid arguments that violate domain rules, preconditions, or semantic constraints.
- **Example:** Querying a SQL table for a column name `customer_zipcode` that does not exist in schema.

### 2.4 `missing_required_parameter` (Missing Required Parameter)
- **Phase:** `tool_invocation`
- **Definition:** The agent omits mandatory fields required by the tool API definition.
- **Example:** Calling `send_email(body="hello")` without the mandatory `to_address` parameter.

---

## 3. Environmental Failure Family (`environmental`)

### 3.1 `tool_execution_error` (Tool Runtime Crash / Timeout)
- **Phase:** `tool_invocation`
- **Definition:** The tool endpoint fails due to internal server errors (500), network timeouts, or runtime crashes outside agent control.
- **Example:** HTTP 504 Gateway Timeout from external third-party API.

### 3.2 `environment_state_inconsistency` (State Inconsistency / Race Condition)
- **Phase:** `tool_invocation`, `tool_result_interpretation`
- **Definition:** External environment state changes concurrently or behaves non-deterministically.
- **Example:** A product was in-stock at step 2, but out-of-stock when purchased at step 4.

### 3.3 `unexpected_observation` (Unexpected Observation Format)
- **Phase:** `tool_result_interpretation`
- **Definition:** Tool output is garbled, corrupted, or violates expected return schema.
- **Example:** Tool returned a raw HTML error stack trace instead of expected JSON payload.

---

## 4. Recovery Failure Family (`recovery`)

### 4.1 `unhandled_tool_error` (Unhandled Tool Error)
- **Phase:** `recovery_adaptation`, `tool_result_interpretation`
- **Definition:** The agent receives an explicit tool error message but ignores it and proceeds as if it succeeded.
- **Example:** Tool output: `Error: File not found`. Agent: `Proceeding to parse file content...`

### 4.2 `repetitive_failing_action` (Repetitive Failing Action / Infinite Loop)
- **Phase:** `recovery_adaptation`, `tool_invocation`
- **Definition:** The agent retries the exact same failing tool call with identical arguments repeatedly.
- **Example:** Retrying `get_user(id=-1)` 5 times in a loop after receiving 404 errors.

### 4.3 `premature_termination` (Premature Termination)
- **Phase:** `recovery_adaptation`, `final_synthesis`
- **Definition:** The agent gives up or terminates early when viable alternative tools or recovery actions remained.
- **Example:** Giving up after one search query failed without attempting keyword synonyms.

---

## 5. Synthesis Failure Family (`synthesis`)

### 5.1 `unsupported_conclusion` (Unsupported Final Conclusion)
- **Phase:** `final_synthesis`
- **Definition:** The final answer makes assertions unsupported by or directly contradicting collected tool observations.
- **Example:** Tool reported `Revenue: $10M`, agent concluded `Revenue was $100M`.

### 5.2 `evidence_omission` (Evidence Omission)
- **Phase:** `final_synthesis`
- **Definition:** Correct evidence was retrieved by tools but omitted from the final answer required by the user.
- **Example:** User requested tracking ID and delivery date; agent provided only the delivery date.

### 5.3 `constraint_violation` (Output Constraint Violation)
- **Phase:** `final_synthesis`
- **Definition:** Final output violates explicit formatting, structural, or length constraints.
- **Example:** User specified `Output in pure JSON`, agent responded in conversational Markdown.
