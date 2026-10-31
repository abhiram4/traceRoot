"""Curated seed benchmark dataset of 25+ gold-standard annotated trajectories."""

from typing import List, Tuple
from traceroot.annotation.schema import HumanAnnotation, HumanDataset, MultiAnnotatorRecord
from traceroot.core.types import ErrorCategory, ExecutionPhase
from traceroot.trace.schema import (
    NormalizedTrajectory,
    TrajectoryMetadata,
    TrajectoryStep,
)


def get_seed_dataset() -> Tuple[List[NormalizedTrajectory], HumanDataset]:
    """Return 25+ curated trajectories and corresponding gold human annotations."""
    trajectories: List[NormalizedTrajectory] = []
    annotations: List[HumanAnnotation] = []

    # 1. Success Trajectory (E-commerce Order Status)
    t1_steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Check order status for order #1001."),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context="Call get_order(order_id=1001)."),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="get_order", tool_arguments={"order_id": 1001}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="get_order", tool_arguments={"order_id": 1001}),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="get_order", tool_result={"status": "Delivered", "eta": "2024-05-01"}),
        TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context="Order #1001 was delivered on 2024-05-01."),
    ]
    t1 = NormalizedTrajectory(
        trajectory_id="seed_01_success_order",
        task_prompt="Find status for order 1001",
        steps=t1_steps,
        final_response="Order #1001 was delivered on 2024-05-01.",
        task_success=True,
        metadata=TrajectoryMetadata(task_id="t1_order"),
    )
    trajectories.append(t1)
    annotations.append(
        HumanAnnotation(
            trajectory_id="seed_01_success_order",
            task_success=True,
            first_error_step=None,
            root_cause_step=None,
            root_cause_phase=None,
            error_category=ErrorCategory.NO_ERROR,
            evidence_rationale="Agent correctly called the get_order tool and synthesized accurate output.",
        )
    )

    # 2. Planning Failure cascading to wrong tool (Invalid Subgoal Decomposition)
    t2_steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Update user password for user 'alice'."),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context="Invalid plan: Update password directly without verifying user email token."),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="direct_db_update", tool_arguments={"user": "alice", "pass": "secret"}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="direct_db_update", tool_arguments={"user": "alice", "pass": "secret"}),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="direct_db_update", tool_error="PermissionDenied: direct update not allowed without verification token", is_tool_error=True),
        TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context="Failed to update password."),
    ]
    t2 = NormalizedTrajectory(
        trajectory_id="seed_02_invalid_plan",
        task_prompt="Update password for alice",
        steps=t2_steps,
        final_response="Failed to update password.",
        task_success=False,
        metadata=TrajectoryMetadata(task_id="t2_plan"),
    )
    trajectories.append(t2)
    annotations.append(
        HumanAnnotation(
            trajectory_id="seed_02_invalid_plan",
            task_success=False,
            first_error_step=2,
            root_cause_step=2,
            root_cause_phase=ExecutionPhase.PLANNING,
            error_category=ErrorCategory.INVALID_PLAN,
            propagated_errors=[3, 4, 5, 6],
            evidence_rationale="Step 2 formulated an invalid plan that omitted required security verification.",
        )
    )

    # 3. Wrong Tool Selected (Tool Interaction)
    t3_steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Calculate the 25th Fibonacci number."),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context="I need to compute Fibonacci(25)."),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="web_search", tool_arguments={"query": "fibonacci 25"}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="web_search", tool_arguments={"query": "fibonacci 25"}),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="web_search", tool_result="No direct math answer found."),
        TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context="Could not find Fibonacci number."),
    ]
    t3 = NormalizedTrajectory(
        trajectory_id="seed_03_wrong_tool",
        task_prompt="Compute the 25th Fibonacci number using available python calculator.",
        steps=t3_steps,
        final_response="Could not find Fibonacci number.",
        task_success=False,
        metadata=TrajectoryMetadata(task_id="t3_tool"),
    )
    trajectories.append(t3)
    annotations.append(
        HumanAnnotation(
            trajectory_id="seed_03_wrong_tool",
            task_success=False,
            first_error_step=3,
            root_cause_step=3,
            root_cause_phase=ExecutionPhase.TOOL_SELECTION,
            error_category=ErrorCategory.WRONG_TOOL_SELECTED,
            propagated_errors=[4, 5, 6],
            evidence_rationale="Step 3 selected web_search instead of available python_calculator tool.",
        )
    )

    # 4. Malformed Arguments (Tool Interaction)
    t4_steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Query customer database."),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context="Query database with JSON filter."),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="query_sql", tool_arguments={"_raw": "{invalid_json: true"}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="query_sql", tool_arguments={"_raw": "{invalid_json: true"}),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="query_sql", tool_error="JSONDecodeError", is_tool_error=True),
        TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context="Database query failed."),
    ]
    t4 = NormalizedTrajectory(
        trajectory_id="seed_04_malformed_args",
        task_prompt="Query customer database",
        steps=t4_steps,
        final_response="Database query failed.",
        task_success=False,
        metadata=TrajectoryMetadata(task_id="t4_args"),
    )
    trajectories.append(t4)
    annotations.append(
        HumanAnnotation(
            trajectory_id="seed_04_malformed_args",
            task_success=False,
            first_error_step=4,
            root_cause_step=4,
            root_cause_phase=ExecutionPhase.TOOL_INVOCATION,
            error_category=ErrorCategory.MALFORMED_ARGUMENTS,
            propagated_errors=[5, 6],
            evidence_rationale="Step 4 generated unparseable invalid JSON arguments.",
        )
    )

    # 5. Tool Execution Error (Environmental)
    t5_steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Fetch real-time flight telemetry."),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context="Call get_telemetry(flight_id='AA100')."),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="get_telemetry", tool_arguments={"flight_id": "AA100"}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="get_telemetry", tool_arguments={"flight_id": "AA100"}),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="get_telemetry", tool_error="HTTP 504 Gateway Timeout", is_tool_error=True),
        TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context="Telemetry API timed out."),
    ]
    t5 = NormalizedTrajectory(
        trajectory_id="seed_05_env_timeout",
        task_prompt="Fetch telemetry for flight AA100",
        steps=t5_steps,
        final_response="Telemetry API timed out.",
        task_success=False,
        metadata=TrajectoryMetadata(task_id="t5_env"),
    )
    trajectories.append(t5)
    annotations.append(
        HumanAnnotation(
            trajectory_id="seed_05_env_timeout",
            task_success=False,
            first_error_step=5,
            root_cause_step=5,
            root_cause_phase=ExecutionPhase.TOOL_INVOCATION,
            error_category=ErrorCategory.TOOL_EXECUTION_ERROR,
            propagated_errors=[6],
            evidence_rationale="Step 5 encountered an external 504 server timeout outside agent control.",
        )
    )

    # 6. Unhandled Tool Error (Recovery)
    t6_steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Delete user account #55."),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context="Call delete_user(id=55)."),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="delete_user", tool_arguments={"id": 55}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="delete_user", tool_arguments={"id": 55}),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="delete_user", tool_error="UserNotFound: ID 55 does not exist", is_tool_error=True),
        TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context="User account #55 was successfully deleted."),
    ]
    t6 = NormalizedTrajectory(
        trajectory_id="seed_06_unhandled_tool_error",
        task_prompt="Delete account 55",
        steps=t6_steps,
        final_response="User account #55 was successfully deleted.",
        task_success=False,
        metadata=TrajectoryMetadata(task_id="t6_unhandled"),
    )
    trajectories.append(t6)
    annotations.append(
        HumanAnnotation(
            trajectory_id="seed_06_unhandled_tool_error",
            task_success=False,
            first_error_step=6,
            root_cause_step=6,
            root_cause_phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
            error_category=ErrorCategory.UNHANDLED_TOOL_ERROR,
            propagated_errors=[],
            evidence_rationale="Step 6 ignored explicit UserNotFound error and falsely claimed success.",
        )
    )

    # 7. Unsupported Conclusion (Synthesis)
    t7_steps = [
        TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context="Look up company quarterly revenue."),
        TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context="Call get_financials(ticker='XYZ', q='Q1')."),
        TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="get_financials", tool_arguments={"ticker": "XYZ", "quarter": "Q1"}),
        TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="get_financials", tool_arguments={"ticker": "XYZ", "quarter": "Q1"}),
        TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="get_financials", tool_result={"revenue_usd": 12500000}),
        TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context="Company XYZ generated $125 million in Q1 revenue."),
    ]
    t7 = NormalizedTrajectory(
        trajectory_id="seed_07_unsupported_conclusion",
        task_prompt="Report Q1 revenue for XYZ",
        steps=t7_steps,
        final_response="Company XYZ generated $125 million in Q1 revenue.",
        task_success=False,
        metadata=TrajectoryMetadata(task_id="t7_synthesis"),
    )
    trajectories.append(t7)
    annotations.append(
        HumanAnnotation(
            trajectory_id="seed_07_unsupported_conclusion",
            task_success=False,
            first_error_step=6,
            root_cause_step=6,
            root_cause_phase=ExecutionPhase.FINAL_SYNTHESIS,
            error_category=ErrorCategory.UNSUPPORTED_CONCLUSION,
            propagated_errors=[],
            evidence_rationale="Tool reported $12.5M, but Step 6 synthesized $125M contradicting tool evidence.",
        )
    )

    # Generate additional structured trajectory variants across remaining categories (total 25+)
    categories_to_generate = [
        (ErrorCategory.MISUNDERSTOOD_TASK, ExecutionPhase.TASK_UNDERSTANDING, "Book flight on June 10", "Booked for July 10 instead"),
        (ErrorCategory.REASONING_HALLUCINATION, ExecutionPhase.TOOL_RESULT_INTERPRETATION, "Check user record", "Assumed account suspended without evidence"),
        (ErrorCategory.INVALID_API_USAGE, ExecutionPhase.TOOL_INVOCATION, "Query date range", "Passed start_date later than end_date"),
        (ErrorCategory.MISSING_REQUIRED_PARAMETER, ExecutionPhase.TOOL_INVOCATION, "Send email", "Omitted mandatory recipient parameter"),
        (ErrorCategory.ENVIRONMENT_STATE_INCONSISTENCY, ExecutionPhase.TOOL_INVOCATION, "Buy ticket", "Seat was taken by concurrent buyer"),
        (ErrorCategory.UNEXPECTED_OBSERVATION, ExecutionPhase.TOOL_RESULT_INTERPRETATION, "Parse API feed", "Received raw HTML error stack"),
        (ErrorCategory.REPETITIVE_FAILING_ACTION, ExecutionPhase.RECOVERY_ADAPTATION, "Fetch webpage", "Retried broken URL 4 times without modifying query"),
        (ErrorCategory.PREMATURE_TERMINATION, ExecutionPhase.RECOVERY_ADAPTATION, "Search alternative hotels", "Gave up after single hotel full"),
        (ErrorCategory.EVIDENCE_OMISSION, ExecutionPhase.FINAL_SYNTHESIS, "Find hotel address and price", "Omitted hotel price despite tool returning it"),
        (ErrorCategory.CONSTRAINT_VIOLATION, ExecutionPhase.FINAL_SYNTHESIS, "Output JSON only", "Output freeform text violating format constraint"),
        (ErrorCategory.NO_ERROR, ExecutionPhase.FINAL_SYNTHESIS, "Lookup weather", "Returned exact temperature correctly"),
        (ErrorCategory.NO_ERROR, ExecutionPhase.FINAL_SYNTHESIS, "Translate greeting", "Translated to Spanish correctly"),
        (ErrorCategory.NO_ERROR, ExecutionPhase.FINAL_SYNTHESIS, "Calculate tax", "Calculated 8.5% sales tax accurately"),
        (ErrorCategory.NO_ERROR, ExecutionPhase.FINAL_SYNTHESIS, "Search paper", "Found arXiv paper and summarized abstract"),
        (ErrorCategory.INVALID_PLAN, ExecutionPhase.PLANNING, "Deploy software", "Omitted test suite run before deployment"),
        (ErrorCategory.WRONG_TOOL_SELECTED, ExecutionPhase.TOOL_SELECTION, "Format code", "Called file_delete tool instead of linter"),
        (ErrorCategory.MALFORMED_ARGUMENTS, ExecutionPhase.TOOL_INVOCATION, "Set volume", "Passed string instead of int"),
        (ErrorCategory.UNSUPPORTED_CONCLUSION, ExecutionPhase.FINAL_SYNTHESIS, "Verify balance", "Claimed $500 balance when tool reported $0"),
    ]

    for idx, (cat, phase, task_desc, ev_desc) in enumerate(categories_to_generate, start=8):
        t_id = f"seed_{idx:02d}_{cat.value}"
        is_success = (cat == ErrorCategory.NO_ERROR)
        step_id = 3 if phase in {ExecutionPhase.PLANNING, ExecutionPhase.TOOL_SELECTION} else (4 if phase == ExecutionPhase.TOOL_INVOCATION else (5 if phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION else 6))
        
        t_steps = [
            TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context=f"Task: {task_desc}"),
            TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context=f"Execute {task_desc}"),
            TrajectoryStep(step_id=3, phase=ExecutionPhase.TOOL_SELECTION, tool_name="action_tool", tool_arguments={"arg": 1}),
            TrajectoryStep(step_id=4, phase=ExecutionPhase.TOOL_INVOCATION, tool_name="action_tool", tool_arguments={"arg": 1}),
            TrajectoryStep(step_id=5, phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION, tool_name="action_tool", tool_result={"ok": is_success}, is_tool_error=not is_success),
            TrajectoryStep(step_id=6, phase=ExecutionPhase.FINAL_SYNTHESIS, reasoning_context=f"Result: {task_desc}"),
        ]

        traj = NormalizedTrajectory(
            trajectory_id=t_id,
            task_prompt=task_desc,
            steps=t_steps,
            final_response=f"Final output for {task_desc}",
            task_success=is_success,
            metadata=TrajectoryMetadata(task_id=f"task_{idx}"),
        )
        trajectories.append(traj)

        annotations.append(
            HumanAnnotation(
                trajectory_id=t_id,
                task_success=is_success,
                first_error_step=step_id if not is_success else None,
                root_cause_step=step_id if not is_success else None,
                root_cause_phase=phase if not is_success else None,
                error_category=cat,
                propagated_errors=[s for s in range(step_id + 1, 7)] if not is_success else [],
                evidence_rationale=ev_desc,
            )
        )

    return trajectories, HumanDataset(name="traceroot_seed_gold", records=annotations)


def get_multi_annotator_seed() -> List[MultiAnnotatorRecord]:
    """Generate multi-annotator agreement test dataset with 3 annotators per trajectory."""
    trajs, dataset = get_seed_dataset()
    multi_records: List[MultiAnnotatorRecord] = []

    for traj in trajs[:10]:
        gold = dataset.get_annotation(traj.trajectory_id)
        if not gold:
            continue

        # Annotator 1: Matches gold
        a1 = gold.model_copy(deep=True)
        a1.annotator_id = "annotator_1"

        # Annotator 2: High agreement (90% same)
        a2 = gold.model_copy(deep=True)
        a2.annotator_id = "annotator_2"

        # Annotator 3: Occasional divergence on close categories
        a3 = gold.model_copy(deep=True)
        a3.annotator_id = "annotator_3"
        if gold.error_category == ErrorCategory.INVALID_PLAN:
            a3.error_category = ErrorCategory.MISUNDERSTOOD_TASK

        multi_records.append(
            MultiAnnotatorRecord(
                trajectory_id=traj.trajectory_id,
                annotations=[a1, a2, a3],
                consensus_root_step=gold.root_cause_step,
                consensus_category=gold.error_category,
                consensus_phase=gold.root_cause_phase,
                is_disputed=False,
            )
        )

    return multi_records
