"""Configuration models and loader for TraceRoot."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file with override=True to ensure workspace keys take precedence
load_dotenv(override=True)

from traceroot.core.types import AggregationStrategy, JudgeType


class JudgeConfig(BaseModel):
    """Configuration for an individual judge model."""
    name: str = "default_judge"
    judge_type: JudgeType = JudgeType.MOCK
    provider: str = "mock"  # "openai", "gemini", "anthropic", "mock", "heuristic"
    model: str = "mock-judge-v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 60.0
    reliability_weight: float = 1.0
    prompt_template: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class EnsembleConfig(BaseModel):
    """Configuration for ensemble judging."""
    judges: List[JudgeConfig] = Field(default_factory=lambda: [JudgeConfig()])
    aggregation_strategy: AggregationStrategy = AggregationStrategy.CONFIDENCE_WEIGHTED
    min_agreement_threshold: float = 0.6
    enable_disagreement_analysis: bool = True


class CalibrationConfig(BaseModel):
    """Configuration for confidence calibration and reliability metrics."""
    method: str = "temperature_scaling"  # "temperature_scaling", "platt", "isotonic"
    n_bins: int = 10
    bin_strategy: str = "uniform"  # "uniform", "quantile"
    val_split_ratio: float = 0.3
    random_seed: int = 42


class ReportingConfig(BaseModel):
    """Configuration for failure attribution reporting."""
    default_format: str = "markdown"  # "markdown", "html", "terminal", "json"
    include_propagation_graph: bool = True
    include_evidence: bool = True
    include_recommendations: bool = True
    output_dir: str = "./reports"


class TraceRootConfig(BaseSettings):
    """Global configuration for TraceRoot."""
    project_name: str = "TraceRoot"
    version: str = "0.1.0"
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    data_dir: str = "./data"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="TRACEROOT_",
        arbitrary_types_allowed=True,
    )


def load_config(config_path: Optional[str | Path] = None) -> TraceRootConfig:
    """Load configuration from a YAML file or return defaults."""
    if not config_path:
        # Check standard default locations
        default_paths = [
            Path("configs/default_config.yaml"),
            Path("traceroot.yaml"),
            Path(".traceroot.yaml"),
        ]
        for p in default_paths:
            if p.exists():
                config_path = p
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if "judges" in data and "ensemble" not in data:
                data = {"ensemble": data}
            return TraceRootConfig(**data)

    return TraceRootConfig()
