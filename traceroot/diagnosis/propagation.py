"""Failure propagation graph modeling using NetworkX."""

import json
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
from pydantic import BaseModel, Field

from traceroot.core.types import (
    ErrorCategory,
    ExecutionPhase,
    PropagationEdgeType,
    PropagationRelation,
)
from traceroot.diagnosis.judge import JudgeDiagnosis, StepAssessment
from traceroot.trace.schema import NormalizedTrajectory, TrajectoryStep


class PropagationNodeData(BaseModel):
    """Metadata attributes associated with a node in the propagation graph."""
    step_id: int
    phase: ExecutionPhase
    relation: PropagationRelation
    error_category: ErrorCategory
    is_error: bool
    confidence: float
    summary: str


class PropagationEdgeData(BaseModel):
    """Metadata attributes associated with a directed propagation edge."""
    source_step_id: int
    target_step_id: int
    edge_type: PropagationEdgeType
    description: str


class PropagationGraph:
    """Directed Acyclic Graph (DAG) representing causal failure cascades across a trajectory."""

    def __init__(self, trajectory_id: str):
        self.trajectory_id = trajectory_id
        self.graph = nx.DiGraph()

    def add_step_node(self, node_data: PropagationNodeData) -> None:
        """Add a step vertex with its diagnosis attributes."""
        self.graph.add_node(node_data.step_id, **node_data.model_dump())

    def add_propagation_edge(self, edge_data: PropagationEdgeData) -> None:
        """Add a directed causal edge between two steps."""
        self.graph.add_edge(
            edge_data.source_step_id,
            edge_data.target_step_id,
            **edge_data.model_dump(),
        )

    @property
    def root_cause_nodes(self) -> List[int]:
        """Return all nodes identified as primary root causes."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("relation") == PropagationRelation.ROOT_CAUSE.value or d.get("relation") == PropagationRelation.ROOT_CAUSE
        ]

    @property
    def propagated_nodes(self) -> List[int]:
        """Return all downstream nodes corrupted by propagation."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get("relation") == PropagationRelation.PROPAGATED_CONSEQUENCE.value or d.get("relation") == PropagationRelation.PROPAGATED_CONSEQUENCE
        ]

    @property
    def cascade_depth(self) -> int:
        """Compute the maximum length of a failure cascade path."""
        roots = self.root_cause_nodes
        if not roots:
            return 0
        max_d = 0
        for r in roots:
            descendants = nx.descendants(self.graph, r)
            subgraph = self.graph.subgraph(descendants | {r})
            if nx.is_directed_acyclic_graph(subgraph):
                max_d = max(max_d, nx.dag_longest_path_length(subgraph))
            else:
                lengths = nx.single_source_shortest_path_length(self.graph, r)
                if lengths:
                    max_d = max(max_d, max(lengths.values()))
        return max_d

    @property
    def blast_radius(self) -> float:
        """Calculate the proportion of total steps contaminated by the failure."""
        total = self.graph.number_of_nodes()
        if total == 0:
            return 0.0
        prop_count = len(self.root_cause_nodes) + len(self.propagated_nodes)
        return min(1.0, prop_count / total)

    def to_mermaid(self) -> str:
        """Render the propagation DAG as a Mermaid flowchart diagram."""
        lines = ["graph TD"]
        # Nodes
        for n, d in sorted(self.graph.nodes(data=True)):
            raw_phase = d.get("phase", "step")
            phase_val = raw_phase.value if hasattr(raw_phase, "value") else str(raw_phase)
            rel = d.get("relation", "normal")
            rel_val = rel.value if hasattr(rel, "value") else str(rel)
            raw_cat = d.get("error_category", "no_error")
            cat_val = raw_cat.value if hasattr(raw_cat, "value") else str(raw_cat)
            
            if rel_val in {PropagationRelation.ROOT_CAUSE.value, "root_cause"}:
                style = ":::rootNode"
                label = f"Step {n}: {phase_val}<br/><b>ROOT: {cat_val}</b>"
            elif rel_val in {PropagationRelation.PROPAGATED_CONSEQUENCE.value, "propagated_consequence"}:
                style = ":::propNode"
                label = f"Step {n}: {phase_val}<br/>Propagated"
            else:
                style = ":::normalNode"
                label = f"Step {n}: {phase_val}"

            lines.append(f'    S{n}["{label}"]{style}')

        # Edges
        for u, v, d in self.graph.edges(data=True):
            raw_e = d.get("edge_type", "causal")
            e_val = raw_e.value if hasattr(raw_e, "value") else str(raw_e)
            lines.append(f'    S{u} -->|"{e_val}"| S{v}')

        # Styles
        lines.append("    classDef rootNode fill:#ff4d4f,stroke:#cf1322,stroke-width:2px,color:#fff;")
        lines.append("    classDef propNode fill:#ffa39e,stroke:#f5222d,stroke-width:1px,color:#000;")
        lines.append("    classDef normalNode fill:#e6f7ff,stroke:#91d5ff,stroke-width:1px,color:#000;")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the propagation graph structure."""
        return {
            "trajectory_id": self.trajectory_id,
            "root_causes": self.root_cause_nodes,
            "propagated_steps": self.propagated_nodes,
            "cascade_depth": self.cascade_depth,
            "blast_radius": self.blast_radius,
            "nodes": [d for _, d in sorted(self.graph.nodes(data=True))],
            "edges": [d for _, _, d in self.graph.edges(data=True)],
        }


class PropagationGraphBuilder:
    """Constructs a typed failure propagation DAG from a trajectory and judge diagnosis."""

    @classmethod
    def build(
        cls,
        trajectory: NormalizedTrajectory,
        diagnosis: JudgeDiagnosis,
    ) -> PropagationGraph:
        """Construct the propagation graph combining data-flow analysis with judge citations."""
        pg = PropagationGraph(trajectory.trajectory_id)
        assessment_map = {sa.step_id: sa for sa in diagnosis.step_assessments}

        # 1. Add all nodes
        for step in trajectory.steps:
            sa = assessment_map.get(step.step_id)
            is_err = sa.is_error if sa else step.is_tool_error
            cat = sa.error_category if sa else ErrorCategory.NO_ERROR
            rel = sa.relation if sa else (PropagationRelation.ROOT_CAUSE if is_err else PropagationRelation.NORMAL)
            conf = sa.confidence if sa else 1.0

            node_data = PropagationNodeData(
                step_id=step.step_id,
                phase=step.phase,
                relation=rel,
                error_category=cat,
                is_error=is_err,
                confidence=conf,
                summary=step.to_readable_summary(),
            )
            pg.add_step_node(node_data)

        # 2. Add sequential execution baseline edges
        for i in range(len(trajectory.steps) - 1):
            curr_s = trajectory.steps[i]
            next_s = trajectory.steps[i + 1]
            
            # Determine edge type
            edge_type = PropagationEdgeType.CAUSAL_DEPENDENCY
            desc = "Sequential execution flow"

            # Check if this is data-flow between tool call and result
            if curr_s.phase == ExecutionPhase.TOOL_INVOCATION and next_s.phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION:
                edge_type = PropagationEdgeType.DATA_FLOW
                desc = "Tool invocation to observation data flow"

            # Check if previous was an error creating state pollution
            curr_sa = assessment_map.get(curr_s.step_id)
            if curr_sa and curr_sa.relation in {PropagationRelation.ROOT_CAUSE, PropagationRelation.PROPAGATED_CONSEQUENCE}:
                edge_type = PropagationEdgeType.STATE_POLLUTION
                desc = "Contaminated state propagation"

            # Check if recovery attempt
            if next_s.phase == ExecutionPhase.RECOVERY_ADAPTATION:
                edge_type = PropagationEdgeType.RECOVERY_ATTEMPT
                desc = "Attempting error recovery"

            pg.add_propagation_edge(
                PropagationEdgeData(
                    source_step_id=curr_s.step_id,
                    target_step_id=next_s.step_id,
                    edge_type=edge_type,
                    description=desc,
                )
            )

        # 3. Add explicit causal parent cross-edges from judge assessment if specified
        for sa in diagnosis.step_assessments:
            if sa.causal_parent_step and sa.causal_parent_step != sa.step_id - 1:
                pg.add_propagation_edge(
                    PropagationEdgeData(
                        source_step_id=sa.causal_parent_step,
                        target_step_id=sa.step_id,
                        edge_type=PropagationEdgeType.CAUSAL_DEPENDENCY,
                        description=f"Judge identified causal dependency: {sa.evidence}",
                    )
                )

        return pg
