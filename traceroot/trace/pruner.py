"""Hierarchical trace pruning and progressive zoom-in token compressor."""

import json
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


class TrajectoryChunk(BaseModel):
    """A semantic chunk of consecutive trajectory steps."""
    chunk_id: int
    start_step_id: int
    end_step_id: int
    summary: str
    has_anomaly: bool = False
    anomaly_step_ids: List[int] = Field(default_factory=list)


class PrunedTrajectoryBundle(BaseModel):
    """A compressed trajectory representation optimized for LLM judge context windows."""
    original_step_count: int
    pruned_step_count: int
    chunks: List[TrajectoryChunk] = Field(default_factory=list)
    zoomed_steps: List[TrajectoryStep] = Field(default_factory=list)
    zoom_window: Optional[Tuple[int, int]] = None
    compression_ratio: float = 1.0


class HierarchicalTracePruner:
    """Compresses oversized agent logs and implements progressive zoom-in windowing.
    
    1. Summarizes large, healthy tool outputs into 1-line semantic observations.
    2. Groups multi-step traces into high-level chunks.
    3. Preserves full, verbatim token fidelity inside localized anomaly windows.
    """

    MAX_OBSERVATION_CHARS = 300
    CHUNK_SIZE = 5

    @classmethod
    def prune_observation(cls, obs: Any, max_chars: int = MAX_OBSERVATION_CHARS) -> Any:
        """Condense long tool observation strings into clean summaries."""
        if obs is None:
            return None

        if isinstance(obs, (dict, list)):
            obs_str = json.dumps(obs)
        else:
            obs_str = str(obs)

        if len(obs_str) <= max_chars:
            return obs

        # Retain head and tail of long observations
        head = obs_str[: max_chars // 2]
        tail = obs_str[- (max_chars // 2):]
        return f"{head} ... [truncated {len(obs_str) - max_chars} chars] ... {tail}"

    @classmethod
    def build_chunk_summary(cls, steps: List[TrajectoryStep]) -> str:
        """Generate a concise 1-2 sentence semantic summary of a step chunk."""
        phases = [s.phase.value for s in steps]
        tool_names = [s.tool_name for s in steps if s.tool_name]
        has_error = any(s.is_tool_error or bool(s.tool_error) for s in steps)

        summary_parts = [f"Phases: {', '.join(dict.fromkeys(phases))}."]
        if tool_names:
            summary_parts.append(f"Tools used: {', '.join(dict.fromkeys(tool_names))}.")
        if has_error:
            summary_parts.append("Contains tool error/exception.")
        else:
            summary_parts.append("Normal execution flow.")

        return " ".join(summary_parts)

    @classmethod
    def compress(
        cls,
        trajectory: NormalizedTrajectory,
        zoom_window: Optional[Tuple[int, int]] = None,
        chunk_size: int = CHUNK_SIZE,
    ) -> Tuple[NormalizedTrajectory, PrunedTrajectoryBundle]:
        """Produce a pruned trajectory and chunk bundle with full fidelity in the zoom window."""
        original_step_count = len(trajectory.steps)
        pruned_steps: List[TrajectoryStep] = []
        chunks: List[TrajectoryChunk] = []

        win_start = zoom_window[0] if zoom_window else 1
        win_end = zoom_window[1] if zoom_window else original_step_count

        # 1. Build step chunks
        for i in range(0, original_step_count, chunk_size):
            chunk_steps = trajectory.steps[i : i + chunk_size]
            c_start = chunk_steps[0].step_id
            c_end = chunk_steps[-1].step_id
            c_errs = [s.step_id for s in chunk_steps if s.is_tool_error or bool(s.tool_error)]

            chunks.append(
                TrajectoryChunk(
                    chunk_id=(i // chunk_size) + 1,
                    start_step_id=c_start,
                    end_step_id=c_end,
                    summary=cls.build_chunk_summary(chunk_steps),
                    has_anomaly=bool(c_errs),
                    anomaly_step_ids=c_errs,
                )
            )

        # 2. Prune individual steps (preserving full verbatim detail inside zoom window)
        zoomed_steps: List[TrajectoryStep] = []
        for step in trajectory.steps:
            in_zoom = win_start <= step.step_id <= win_end

            if in_zoom:
                # Full fidelity verbatim step
                pruned_steps.append(step)
                zoomed_steps.append(step)
            else:
                # Pruned observation step
                pruned_obs = cls.prune_observation(step.tool_result) if step.tool_result is not None else None
                pruned_err = cls.prune_observation(step.tool_error) if step.tool_error is not None else None
                pruned_reasoning = step.reasoning_context
                if pruned_reasoning and len(pruned_reasoning) > 400:
                    pruned_reasoning = pruned_reasoning[:400] + "... [truncated]"

                pruned_step = step.model_copy(
                    update={
                        "tool_result": pruned_obs,
                        "tool_error": pruned_err,
                        "reasoning_context": pruned_reasoning,
                    }
                )
                pruned_steps.append(pruned_step)

        pruned_traj = trajectory.model_copy(update={"steps": pruned_steps})

        bundle = PrunedTrajectoryBundle(
            original_step_count=original_step_count,
            pruned_step_count=len(pruned_steps),
            chunks=chunks,
            zoomed_steps=zoomed_steps,
            zoom_window=(win_start, win_end) if zoom_window else None,
            compression_ratio=round(len(str(pruned_steps)) / max(1, len(str(trajectory.steps))), 3),
        )

        return pruned_traj, bundle
