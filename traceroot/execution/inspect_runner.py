"""Inspect AI runner adapter and live Inspect Scorer plugin."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from traceroot.diagnosis.judge import BaseJudge, MockJudge
from traceroot.trace.normalize import TrajectoryNormalizer
from traceroot.trace.schema import NormalizedTrajectory


class InspectRunnerBridge:
    """Bridges Inspect AI execution runs and log outputs with TraceRoot."""

    @staticmethod
    def load_inspect_logs(log_dir_or_file: Union[str, Path]) -> List[NormalizedTrajectory]:
        """Parse one or multiple Inspect AI .json or .eval log files."""
        p = Path(log_dir_or_file)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {log_dir_or_file}")

        normalizer = TrajectoryNormalizer()
        trajectories: List[NormalizedTrajectory] = []

        files = [p] if p.is_file() else list(p.glob("*.json")) + list(p.glob("*.eval"))

        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if "samples" in data and isinstance(data["samples"], list):
                        for sample in data["samples"]:
                            sample_bundle = {
                                "eval": data.get("eval", {}),
                                "samples": [sample],
                            }
                            traj = normalizer.normalize(sample_bundle, framework="inspect_ai")
                            trajectories.append(traj)
                    else:
                        traj = normalizer.normalize(data, framework="inspect_ai")
                        trajectories.append(traj)
            except Exception:
                continue

        return trajectories

    @staticmethod
    def create_inspect_scorer(judge: Optional[BaseJudge] = None) -> Callable:
        """Create an Inspect AI compatible scorer for online evaluation."""
        eval_judge = judge or MockJudge()
        normalizer = TrajectoryNormalizer()

        def traceroot_scorer(state: Any, target: Any) -> Dict[str, Any]:
            """Online Inspect AI scorer function."""
            # Extract sample from Inspect state
            sample_data = {
                "id": getattr(state, "sample_id", "inspect_sample"),
                "input": getattr(state, "input", ""),
                "target": str(target),
                "messages": [
                    m.model_dump() if hasattr(m, "model_dump") else m
                    for m in getattr(state, "messages", [])
                ],
                "output": {
                    "choices": [{"message": {"content": getattr(state, "output", "")}}]
                },
            }
            traj = normalizer.normalize({"samples": [sample_data]}, framework="inspect_ai")
            diagnosis = eval_judge.diagnose(traj)

            return {
                "value": 1.0 if not diagnosis.failure_detected else 0.0,
                "explanation": diagnosis.summary_rationale,
                "metadata": {
                    "root_cause_step": diagnosis.root_cause_step_id,
                    "root_cause_category": diagnosis.root_cause_category.value,
                    "confidence": diagnosis.confidence,
                },
            }

        return traceroot_scorer
