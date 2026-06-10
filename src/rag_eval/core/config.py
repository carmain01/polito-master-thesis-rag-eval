"""Configuration management for the evaluation framework."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM judge."""

    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.).")
    model: str = Field(default="gpt-4o", description="Model identifier.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    api_key: str = Field(default="", description="API key (prefer env var).")


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevance"],
        description="List of metric names to run.",
    )
    dataset_path: str = Field(default="", description="Path to the evaluation dataset.")
    output_dir: str = Field(default="./output", description="Directory for evaluation outputs.")
    batch_size: int = Field(default=10, gt=0)
    max_concurrency: int = Field(default=5, gt=0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
