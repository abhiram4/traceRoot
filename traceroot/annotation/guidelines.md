# TraceRoot Human Annotation Protocol & Guidelines

This document establishes standard operating procedures for annotating tool-using agent trajectories for failure localization and error taxonomy classification.

---

## 1. Guiding Principles

### 1.1 What Counts as an Error?
A trajectory step is considered **erroneous** if, given the information available to the agent at that exact point in time, the action or thought **materially decreases the probability of successfully completing the task**.

- **Not an Error:** Choosing a suboptimal but valid search query that still succeeds.
- **Is an Error:** Omitting a required filter, ignoring an error message, or executing an action with invalid syntax.

### 1.2 Root Cause vs. Propagated Failure
- **Root Cause ($s_{root}$):** The **earliest** incorrect decision or action without which the final task failure would likely not have occurred.
- **Propagated Failure:** A downstream step that failed primarily because it was operating on invalid state, incorrect tool output, or flawed assumptions created by an earlier error.
- **Independent Failure:** A failure that would have occurred even if all prior steps had been executed perfectly.

---

## 2. Step-by-Step Annotation Procedure

For each assigned trajectory:

1. **Review Task Specification:**
   - Read the user prompt, constraints, and available tool definitions.
2. **Examine Trajectory Step-by-Step:**
   - Step 1..N: Review the agent's internal thought, tool call, arguments, and tool result.
3. **Determine Task Outcome:**
   - Label `task_success` (`true` / `false`).
4. **Identify First Error vs Root Cause:**
   - Note the first step where behavior visibly diverged from correct execution (`first_error_step`).
   - Trace back to find if an earlier planning or reasoning step causally caused this divergence (`root_cause_step`).
5. **Assign Execution Phase & Error Category:**
   - Select the `root_cause_phase` (e.g. `planning`, `tool_selection`, `tool_invocation`, `synthesis`).
   - Assign the exact `error_category` from the TraceRoot taxonomy.
6. **Tag Propagated Consequence Steps:**
   - List all downstream step numbers that failed as a consequence (`propagated_errors`).
7. **Provide Evidence Rationale:**
   - Write a 1-2 sentence factual explanation citing specific step numbers.
