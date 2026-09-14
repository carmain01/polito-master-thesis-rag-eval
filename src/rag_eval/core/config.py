"""Configuration management for the evaluation framework."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from project root (if it exists)
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for the LLM judge."""

    provider: str = Field(
        default="ollama",
        description="LLM provider: 'openai', 'anthropic', 'google', 'ollama', 'vllm', 'azure'.",
        pattern=r'^(openai|anthropic|google|ollama|vllm|azure)$',
    )
    model: str = Field(
        default="llama3.2",
        description="Model identifier. Default is Ollama's llama3.2 (local, free).",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    api_key: str = Field(
        default="",
        description="API key. If empty, reads from env var (e.g. OPENAI_API_KEY).",
    )
    api_base: str = Field(
        default="",
        description="Custom API base URL (for Ollama, vLLM, Azure, etc.).",
    )
    api_version: str = Field(
        default="2024-06-01",
        description="API version for Azure OpenAI.",
    )
    reasoning_effort: str = Field(
        default="high",
        description="Reasoning effort for reasoning models ('low', 'medium', 'high').",
    )
    timeout: float = Field(default=300.0, gt=0, description="Request timeout in seconds.")
    max_retries: int = Field(default=5, ge=0, description="Max retries on transient errors.")

    def model_post_init(self, __context: Any) -> None:
        if self.provider == "azure" and self.model == "llama3.2":
            azure_model = os.getenv("AZURE_MODELS") or os.getenv("AZURE_MODEL")
            if azure_model:
                self.model = azure_model.strip().strip('"')

    def get_api_key(self) -> str:
        """Resolve API key: explicit value > env var."""
        if self.api_key:
            return self.api_key

        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }
        env_var = env_var_map.get(self.provider, "")
        if env_var:
            return os.getenv(env_var, "").strip().strip('"')
        return ""

    def get_api_base(self) -> str:
        """Resolve API base URL: explicit value > env var > default."""
        if self.api_base:
            return self.api_base

        defaults = {
            "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "vllm": os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
            "azure": os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().strip('"'),
        }
        return defaults.get(self.provider, "")


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    provider: str = Field(
        default="sentence-transformers",
        description="Embedding provider: 'sentence-transformers'.",
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name. Default is a fast, lightweight Sentence Transformers model.",
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size for embedding generation.")


class CacheConfig(BaseModel):
    """Configuration for LLM response caching."""

    enabled: bool = Field(default=False, description="Enable/disable response caching.")
    directory: str = Field(
        default=".cache/rag_eval",
        description="Cache directory path.",
    )
    ttl_seconds: int | None = Field(
        default=None,
        description="Time-to-live in seconds. None = no expiration.",
    )


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
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

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**data)
