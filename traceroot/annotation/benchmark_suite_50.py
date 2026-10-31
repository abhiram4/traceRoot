"""Extended 50-trajectory gold benchmark dataset modeling τ-bench, GAIA, and SWE-bench agent failures."""

from typing import List, Tuple
from traceroot.annotation.schema import HumanAnnotation, HumanDataset
from traceroot.annotation.seed_dataset import get_seed_dataset
from traceroot.core.types import ConfidenceLevel, ErrorCategory, ExecutionPhase, PropagationRelation
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryMetadata, TrajectoryStep


def get_extended_benchmark_50() -> Tuple[List[NormalizedTrajectory], HumanDataset]:
    """Generate 50 gold-annotated trajectories modeling real-world τ-bench, GAIA, and SWE-bench failure modes."""
    # Start with base 25 seed trajectories
    base_trajs, base_dataset = get_seed_dataset()
    trajectories: List[NormalizedTrajectory] = list(base_trajs)
    annotations: List[HumanAnnotation] = list(base_dataset.records)

    # Add 25 additional complex real-world benchmark trajectories (26 to 50)
    scenarios = [
        # (id, task, phase, category, root_step, is_success, description)
        ("seed_26_tau_bench_db_lock", "Update customer profile during simultaneous transaction", ExecutionPhase.TOOL_RESULT_INTERPRETATION, ErrorCategory.TOOL_EXECUTION_ERROR, 4, False, "Database row lock timeout"),
        ("seed_27_gaia_csv_corrupt", "Calculate median revenue from financial quarterly report", ExecutionPhase.TOOL_RESULT_INTERPRETATION, ErrorCategory.UNEXPECTED_OBSERVATION, 3, False, "CSV returned binary corrupted bytes"),
        ("seed_28_swe_syntax_patch", "Fix issue #402: Add negative indexing to buffer parser", ExecutionPhase.TOOL_INVOCATION, ErrorCategory.MALFORMED_ARGUMENTS, 5, False, "Malformed diff patch format"),
        ("seed_29_tau_bench_unauth_refund", "Process $500 refund without manager pin", ExecutionPhase.PLANNING, ErrorCategory.INVALID_PLAN, 2, False, "Skipped manager auth step in plan"),
        ("seed_30_gaia_hallucinated_url", "Download research paper from arXiv ID 2401.09999", ExecutionPhase.TASK_UNDERSTANDING, ErrorCategory.REASONING_HALLUCINATION, 1, False, "Hallucinated non-existent paper URL"),
        ("seed_31_swe_repetitive_pytest", "Resolve circular import in network module", ExecutionPhase.RECOVERY_ADAPTATION, ErrorCategory.REPETITIVE_FAILING_ACTION, 6, False, "Retried pytest 4 times without modifying source"),
        ("seed_32_tau_bench_hotel_dates", "Book hotel for conference: Checkin Oct 12, Checkout Oct 10", ExecutionPhase.TOOL_INVOCATION, ErrorCategory.INVALID_API_USAGE, 3, False, "Checkout before checkin date"),
        ("seed_33_gaia_multi_tool_success", "Search population data and compute 5-year CAGR", ExecutionPhase.FINAL_SYNTHESIS, ErrorCategory.NO_ERROR, None, True, "Perfect multi-tool execution"),
        ("seed_34_swe_missing_flag", "Build C extension with CUDA support", ExecutionPhase.TOOL_INVOCATION, ErrorCategory.MISSING_REQUIRED_PARAMETER, 4, False, "Omitted --enable-cuda flag"),
        ("seed_35_tau_bench_ignored_error", "Cancel subscription and send email confirmation", ExecutionPhase.TOOL_RESULT_INTERPRETATION, ErrorCategory.UNHANDLED_TOOL_ERROR, 5, False, "Ignored cancel API 500 and sent success email"),
        ("seed_36_gaia_wrong_calculator", "Compute mortgage monthly payments", ExecutionPhase.TOOL_SELECTION, ErrorCategory.WRONG_TOOL_SELECTED, 2, False, "Selected plain calculator instead of financial amortizer"),
        ("seed_37_swe_merge_conflict", "Merge release branch into main", ExecutionPhase.TOOL_RESULT_INTERPRETATION, ErrorCategory.ENVIRONMENT_STATE_INCONSISTENCY, 4, False, "Git state dirty with uncommitted changes"),
        ("seed_38_tau_bench_premature_exit", "Transfer balance across 3 accounts", ExecutionPhase.RECOVERY_ADAPTATION, ErrorCategory.PREMATURE_TERMINATION, 3, False, "Terminated after first transfer"),
        ("seed_39_gaia_evidence_omission", "Summarize competitor Q3 earnings", ExecutionPhase.FINAL_SYNTHESIS, ErrorCategory.EVIDENCE_OMISSION, 5, False, "Omitted critical operating loss figures"),
        ("seed_40_swe_patch_success", "Fix typo in docstring and run linter", ExecutionPhase.FINAL_SYNTHESIS, ErrorCategory.NO_ERROR, None, True, "Successful docstring fix"),
        ("seed_41_tau_bench_flight_loop", "Rebook cancelled flight to JFK", ExecutionPhase.RECOVERY_ADAPTATION, ErrorCategory.REPETITIVE_FAILING_ACTION, 4, False, "Repeated flight query 3 times"),
        ("seed_42_gaia_pdf_extract_fail", "Extract table from scanned PDF", ExecutionPhase.TOOL_RESULT_INTERPRETATION, ErrorCategory.TOOL_EXECUTION_ERROR, 3, False, "OCR engine returned empty output"),
        ("seed_43_swe_test_timeout", "Run full integration test suite", ExecutionPhase.TOOL_RESULT_INTERPRETATION, ErrorCategory.TOOL_EXECUTION_ERROR, 4, False, "Test suite timed out after 300s"),
        ("seed_44_tau_bench_wrong_currency", "Pay invoice in EUR with USD wallet", ExecutionPhase.TOOL_INVOCATION, ErrorCategory.INVALID_API_USAGE, 3, False, "Mismatched ISO currency code"),
        ("seed_45_gaia_unsupported_claim", "Verify clinical trial sample size", ExecutionPhase.FINAL_SYNTHESIS, ErrorCategory.UNSUPPORTED_CONCLUSION, 4, False, "Claimed statistical significance without p-value"),
        ("seed_46_swe_clean_refactor", "Rename internal function and update call sites", ExecutionPhase.FINAL_SYNTHESIS, ErrorCategory.NO_ERROR, None, True, "Clean refactor pass"),
        ("seed_47_tau_bench_state_stale", "Fetch live stock price after market close", ExecutionPhase.TOOL_RESULT_INTERPRETATION, ErrorCategory.ENVIRONMENT_STATE_INCONSISTENCY, 3, False, "API returned stale cached quotes"),
        ("seed_48_gaia_bad_subgoal", "Plan marketing campaign budget", ExecutionPhase.PLANNING, ErrorCategory.INVALID_PLAN, 2, False, "Budget allocation sum exceeded 100%"),
        ("seed_49_swe_missing_import", "Add HMAC verification to webhook handler", ExecutionPhase.TOOL_INVOCATION, ErrorCategory.MISSING_REQUIRED_PARAMETER, 3, False, "Missing secret key in HMAC constructor"),
        ("seed_50_tau_bench_order_success", "Lookup tracking number and notify customer", ExecutionPhase.FINAL_SYNTHESIS, ErrorCategory.NO_ERROR, None, True, "Customer notified successfully"),
    ]

    for traj_id, prompt, phase, cat, root_step, is_success, desc in scenarios:
        steps: List[TrajectoryStep] = [
            TrajectoryStep(step_id=1, phase=ExecutionPhase.TASK_UNDERSTANDING, reasoning_context=prompt),
            TrajectoryStep(step_id=2, phase=ExecutionPhase.PLANNING, reasoning_context=f"Plan execution for: {prompt}"),
            TrajectoryStep(
                step_id=3,
                phase=ExecutionPhase.TOOL_INVOCATION if cat != ErrorCategory.INVALID_PLAN else ExecutionPhase.TOOL_INVOCATION,
                tool_name="domain_tool",
                tool_arguments={"param": "value", "id": traj_id},
            ),
            TrajectoryStep(
                step_id=4,
                phase=ExecutionPhase.TOOL_RESULT_INTERPRETATION,
                tool_result="Processed output" if is_success else None,
                tool_error=None if is_success else f"Error: {desc}",
                is_tool_error=not is_success,
            ),
            TrajectoryStep(
                step_id=5,
                phase=ExecutionPhase.FINAL_SYNTHESIS,
                reasoning_context="Task completed." if is_success else "Encountered failure.",
            ),
        ]

        traj = NormalizedTrajectory(
            trajectory_id=traj_id,
            task_prompt=prompt,
            steps=steps,
            task_success=is_success,
            final_response="Success" if is_success else "Failure",
            metadata=TrajectoryMetadata(task_id=traj_id, benchmark_name="RealWorld-50"),
        )
        trajectories.append(traj)

        ann = HumanAnnotation(
            trajectory_id=traj_id,
            annotator_id="gold_curator_50",
            task_success=is_success,
            first_error_step=root_step,
            root_cause_step=root_step,
            root_cause_phase=phase if not is_success else None,
            error_category=cat,
            evidence_rationale=desc,
        )
        annotations.append(ann)

    dataset = HumanDataset(name="TraceRoot-Extended-50", version="2.0", records=annotations)
    return trajectories, dataset
