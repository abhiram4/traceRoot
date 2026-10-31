"""Failure Attribution Report generator for Markdown, HTML, JSON, and terminal output."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import jinja2

from traceroot.core.types import ErrorCategory, ExecutionPhase, PropagationRelation
from traceroot.diagnosis.judge import JudgeDiagnosis
from traceroot.diagnosis.propagation import PropagationGraph, PropagationGraphBuilder
from traceroot.diagnosis.remediation import RemediationGenerator
from traceroot.reporting.visualizer import TerminalVisualizer
from traceroot.taxonomy.categories import get_category_definition
from traceroot.trace.schema import NormalizedTrajectory


class ReportGenerator:
    """Generates structured failure attribution reports in multiple output formats."""

    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        if not template_dir:
            template_dir = Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    def generate_markdown(
        self,
        trajectory: NormalizedTrajectory,
        diagnosis: JudgeDiagnosis,
        propagation_graph: Optional[PropagationGraph] = None,
    ) -> str:
        """Render standard GitHub-flavored Markdown failure attribution report."""
        if not propagation_graph:
            propagation_graph = PropagationGraphBuilder.build(trajectory, diagnosis)

        context = self._build_template_context(trajectory, diagnosis, propagation_graph)
        template = self.jinja_env.get_template("report_markdown.jinja2")
        return template.render(**context)

    def generate_html(
        self,
        trajectory: NormalizedTrajectory,
        diagnosis: JudgeDiagnosis,
        propagation_graph: Optional[PropagationGraph] = None,
    ) -> str:
        """Render rich standalone HTML failure attribution report."""
        if not propagation_graph:
            propagation_graph = PropagationGraphBuilder.build(trajectory, diagnosis)

        context = self._build_template_context(trajectory, diagnosis, propagation_graph)
        template = self.jinja_env.get_template("report_html.jinja2")
        return template.render(**context)

    def generate_json(
        self,
        trajectory: NormalizedTrajectory,
        diagnosis: JudgeDiagnosis,
        propagation_graph: Optional[PropagationGraph] = None,
    ) -> str:
        """Serialize report to structured JSON."""
        if not propagation_graph:
            propagation_graph = PropagationGraphBuilder.build(trajectory, diagnosis)

        data = {
            "trajectory_id": trajectory.trajectory_id,
            "task_outcome": "success" if trajectory.task_success else "failed",
            "diagnosis": diagnosis.model_dump(),
            "propagation_graph": propagation_graph.to_dict(),
            "recommendations": self._get_recommendations(diagnosis.root_cause_category),
        }
        return json.dumps(data, indent=2, default=str)

    def display_terminal(
        self,
        trajectory: NormalizedTrajectory,
        diagnosis: JudgeDiagnosis,
    ) -> None:
        """Print colorized table and diagnostic report directly to the terminal."""
        TerminalVisualizer.display_report(trajectory, diagnosis)

    def save_report(
        self,
        content: str,
        output_path: Union[str, Path],
    ) -> Path:
        """Save report content to disk."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    def _build_template_context(
        self,
        trajectory: NormalizedTrajectory,
        diagnosis: JudgeDiagnosis,
        pg: PropagationGraph,
    ) -> Dict[str, Any]:
        """Construct data dictionary for Jinja2 templates."""
        assessment_map = {sa.step_id: sa for sa in diagnosis.step_assessments}
        steps_summary = []

        for step in trajectory.steps:
            sa = assessment_map.get(step.step_id)
            is_err = sa.is_error if sa else step.is_tool_error
            rel = "Normal"
            if sa and sa.relation == PropagationRelation.ROOT_CAUSE:
                rel = "Root cause"
            elif sa and sa.relation == PropagationRelation.PROPAGATED_CONSEQUENCE:
                rel = "Propagated"

            steps_summary.append({
                "step_id": step.step_id,
                "phase": step.phase.value,
                "is_error": is_err,
                "relationship": rel,
                "evidence": sa.evidence if sa else step.to_readable_summary(),
            })

        cat_defn = get_category_definition(diagnosis.root_cause_category)
        cat_name = cat_defn.name if cat_defn else diagnosis.root_cause_category.value
        recs = cat_defn.recommended_interventions if cat_defn else []

        target_comp = "General Agent Architecture"
        if diagnosis.root_cause_phase == ExecutionPhase.PLANNING:
            target_comp = "Planning / Subgoal Decomposition Engine"
        elif diagnosis.root_cause_phase == ExecutionPhase.TOOL_SELECTION:
            target_comp = "Tool Retrieval & Routing Module"
        elif diagnosis.root_cause_phase == ExecutionPhase.TOOL_INVOCATION:
            target_comp = "Tool Schema Formulator / JSON Validator"
        elif diagnosis.root_cause_phase == ExecutionPhase.TOOL_RESULT_INTERPRETATION:
            target_comp = "Observation Parser / Grounding Guardrail"
        elif diagnosis.root_cause_phase == ExecutionPhase.RECOVERY_ADAPTATION:
            target_comp = "Error Reflection & Recovery Policy"
        elif diagnosis.root_cause_phase == ExecutionPhase.FINAL_SYNTHESIS:
            target_comp = "Synthesis & Citation Verifier"

        remediation_patch = RemediationGenerator.generate(trajectory, diagnosis)

        return {
            "trajectory": trajectory,
            "diagnosis": diagnosis,
            "category_name": cat_name,
            "steps_summary": steps_summary,
            "recommendations": recs,
            "target_component": target_comp,
            "mermaid_chart": pg.to_mermaid() if pg else "",
            "remediation": remediation_patch,
        }

    def _get_recommendations(self, category: ErrorCategory) -> List[str]:
        defn = get_category_definition(category)
        return defn.recommended_interventions if defn else []
