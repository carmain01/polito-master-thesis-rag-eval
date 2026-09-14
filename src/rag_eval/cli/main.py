"""Command Line Interface for the RAG Evaluation framework."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rag_eval.core.config import EvalConfig, LLMConfig
from rag_eval.core.registry import MetricRegistry
from rag_eval.core.types import EvalReport
from rag_eval.datasets.loader import load_dataset
from rag_eval.datasets.synthetic import SyntheticDataGenerator
from rag_eval.metrics.base import BaseMetric
from rag_eval.pipeline.evaluator import Evaluator
from rag_eval.reports.generator import ReportGenerator
from rag_eval.utils.llm import LLMClient

app = typer.Typer(
    name="rag-eval",
    help="Modular framework for evaluating Retrieval-Augmented Generation (RAG) systems.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Global options (--verbose / --quiet)
# ---------------------------------------------------------------------------


class Verbosity(str, Enum):
    verbose = "verbose"
    quiet = "quiet"
    normal = "normal"


def _configure_logging(verbosity: Verbosity) -> None:
    """Map verbosity level to Python logging."""
    level_map = {
        Verbosity.verbose: logging.DEBUG,
        Verbosity.normal: logging.WARNING,
        Verbosity.quiet: logging.CRITICAL,
    }
    logging.basicConfig(
        level=level_map[verbosity],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def _version_callback(value: bool) -> None:
    if value:
        console.print("rag-eval 0.1.0")
        raise typer.Exit()


@app.callback()
def main_callback(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress all output except errors.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", help="Show version and exit.", callback=_version_callback, is_eager=True
        ),
    ] = False,
) -> None:
    """Global options applied before every command."""
    if quiet:
        _configure_logging(Verbosity.quiet)
    elif verbose:
        _configure_logging(Verbosity.verbose)
    else:
        _configure_logging(Verbosity.normal)


# ---------------------------------------------------------------------------
# Smart metric instantiation
# ---------------------------------------------------------------------------


def _needs_param(metric_cls: type[BaseMetric], param_name: str) -> bool:
    """Check whether a metric class __init__ accepts *param_name*."""
    sig = inspect.signature(metric_cls.__init__)
    return param_name in sig.parameters


def _instantiate_metrics(
    metric_names: list[str],
    llm_client: LLMClient,
    eval_config: EvalConfig,
) -> list[BaseMetric]:
    """Instantiate metrics by name, creating heavy clients only when needed."""
    MetricRegistry.discover_metrics()
    embedding_client = None  # lazy — created on first use
    instances: list[BaseMetric] = []

    for name in metric_names:
        # Resolve the class first so we can inspect its signature
        target = name.lower().replace("-", "").replace("_", "")
        metric_cls: type[BaseMetric] | None = None
        for key, cls in MetricRegistry._registry.items():
            if key.lower().replace("-", "").replace("_", "") == target:
                metric_cls = cls
                break
        if metric_cls is None:
            console.print(
                f"[bold red]Unknown metric '{name}'.[/bold red]  "
                f"Run [bold]rag-eval metrics[/bold] to list available metrics."
            )
            raise typer.Exit(1)

        # Build kwargs dynamically based on what the class actually accepts
        kwargs: dict[str, object] = {}
        if _needs_param(metric_cls, "llm_client"):
            kwargs["llm_client"] = llm_client
        if _needs_param(metric_cls, "embed_client"):
            if embedding_client is None:
                from rag_eval.utils.embeddings import EmbeddingClient

                console.print("[dim]Loading embedding model (first use)...[/dim]")
                embedding_client = EmbeddingClient(config=eval_config.embedding)
            kwargs["embed_client"] = embedding_client

        try:
            instances.append(metric_cls(**kwargs))
        except Exception as e:
            console.print(f"[bold red]Failed to create metric '{name}':[/bold red] {e}")
            raise typer.Exit(1)

    return instances


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to the dataset JSON/CSV file.", exists=True),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to YAML configuration file.", exists=True),
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for reports.")
    ] = Path("output"),
    metrics: Annotated[
        str,
        typer.Option(
            "--metrics", "-m", help="Comma-separated metric names (e.g. faithfulness,token_f1)."
        ),
    ] = "",
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Comma-separated output formats: json, html, csv, console. Default: all.",
        ),
    ] = "json,html,csv,console",
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="LLM provider override (openai, azure, ollama, vllm, ...)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="LLM model override (e.g. gpt-5-nano, gpt-4o, llama3.2)."),
    ] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option("--reasoning-effort", help="Reasoning effort override ('low', 'medium', 'high')."),
    ] = None,
    max_concurrency: Annotated[
        int, typer.Option("--max-concurrency", help="Maximum concurrent evaluations.")
    ] = 5,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate config and dataset without running.")
    ] = False,
) -> None:
    """Run evaluation on a dataset using specified metrics."""
    # 1. Load configuration ------------------------------------------------
    if config:
        console.print(f"[dim]Loading configuration from {config}[/dim]")
        eval_config = EvalConfig.from_yaml(config)
    else:
        eval_config = EvalConfig()

    # Apply CLI overrides
    if provider:
        eval_config.llm.provider = provider
    if model:
        eval_config.llm.model = model
    elif eval_config.llm.provider == "azure" and eval_config.llm.model == "llama3.2":
        import os
        azure_model = os.getenv("AZURE_MODELS") or os.getenv("AZURE_MODEL")
        if azure_model:
            eval_config.llm.model = azure_model.strip().strip('"')
    if reasoning_effort:
        eval_config.llm.reasoning_effort = reasoning_effort
    if metrics:
        eval_config.metrics = [m.strip() for m in metrics.split(",")]

    # Parse requested output formats
    formats = {f.strip().lower() for f in format.split(",")}
    valid_formats = {"json", "html", "csv", "console"}
    invalid = formats - valid_formats
    if invalid:
        console.print(
            f"[bold red]Unknown format(s): {', '.join(invalid)}.[/bold red]  Valid: {', '.join(valid_formats)}"
        )
        raise typer.Exit(1)

    # 2. Load dataset ------------------------------------------------------
    console.print(f"[dim]Loading dataset from {dataset}[/dim]")
    try:
        samples = load_dataset(dataset)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        raise typer.Exit(1)

    console.print(f"Loaded [bold]{len(samples)}[/bold] samples to evaluate.")

    # 3. Dry-run exit ------------------------------------------------------
    if dry_run:
        console.print("\n[bold green]✓ Dry-run completed successfully![/bold green]")
        console.print("[dim]Resolved configuration:[/dim]")
        console.print(eval_config.model_dump())
        return

    # 4. Initialize clients & metrics --------------------------------------
    llm_client = LLMClient(config=eval_config.llm, cache_config=eval_config.cache)
    metric_instances = _instantiate_metrics(eval_config.metrics, llm_client, eval_config)

    # 5. Evaluate ----------------------------------------------------------
    console.print("[bold blue]Starting evaluation...[/bold blue]")
    evaluator = Evaluator(metrics=metric_instances)
    report = evaluator.evaluate(samples, max_concurrency=max_concurrency)

    # 6. Generate reports --------------------------------------------------
    output.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[bold green]Generating reports in {output}/...[/bold green]")
    generator = ReportGenerator(report)

    if "console" in formats:
        generator.to_console()
    if "json" in formats:
        generator.to_json(output / "report.json")
        console.print(f" ✓ JSON  → {output / 'report.json'}")
    if "csv" in formats:
        generator.to_csv(output / "report.csv")
        console.print(f" ✓ CSV   → {output / 'report.csv'}")
    if "html" in formats:
        generator.to_html(output / "report.html")
        console.print(f" ✓ HTML  → {output / 'report.html'}")

    console.print("\n[bold green]Evaluation complete![/bold green]")


@app.command()
def generate(
    documents: Annotated[
        Path,
        typer.Option("--documents", "-d", help="Path to source document text file.", exists=True),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output JSON file for synthetic dataset.")
    ] = Path("synthetic_dataset.json"),
    num_samples: Annotated[
        int, typer.Option("--num-samples", "-n", help="Number of questions to generate per chunk.")
    ] = 1,
    question_types: Annotated[
        str,
        typer.Option(
            "--question-types",
            "-q",
            help="Comma-separated types: factual, multi-hop, reasoning, comparative.",
        ),
    ] = "factual",
    difficulty: Annotated[
        str, typer.Option("--difficulty", help="Difficulty level: easy, medium, hard.")
    ] = "medium",
    provider: Annotated[
        str, typer.Option("--provider", "-p", help="LLM provider (e.g. ollama, openai).")
    ] = "ollama",
    model: Annotated[
        str, typer.Option("--model", "-m", help="LLM model to use.")
    ] = "qwen2.5:7b",
    max_chunks: Annotated[
        int, typer.Option("--max-chunks", help="Maximum number of chunks to process (0 for unlimited).")
    ] = 10,
    max_concurrency: Annotated[
        int, typer.Option("--max-concurrency", help="Maximum concurrent LLM requests.")
    ] = 3,
) -> None:
    """Generate a synthetic QA dataset from source documents."""
    if documents.suffix.lower() == ".pdf":
        console.print("[bold red]Error:[/bold red] The generate command currently expects a raw text file (.txt), not a PDF. Please extract the text from your PDF into a .txt file first.")
        raise typer.Exit(1)
        
    try:
        text = documents.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        console.print("[bold red]Error:[/bold red] Could not read the file as UTF-8 text. Make sure you are passing a plain text file (.txt).")
        raise typer.Exit(1)
        
    q_types = [t.strip() for t in question_types.split(",")]

    config = LLMConfig(provider=provider, model=model)
    llm_client = LLMClient(config=config)
    gen = SyntheticDataGenerator(llm_client=llm_client)
    console.print(f"[bold blue]Generating synthetic dataset using {provider} ({model})...[/bold blue]")

    samples = asyncio.run(
        gen.generate_qa_pairs(
            text=text,
            num_questions_per_chunk=num_samples,
            question_types=q_types,
            difficulty=difficulty,
            max_chunks=max_chunks if max_chunks > 0 else None,
            max_concurrency=max_concurrency,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump([s.model_dump() for s in samples], f, indent=2)

    console.print(
        f"[bold green]✓ Generated {len(samples)} synthetic samples → {output}[/bold green]"
    )


@app.command()
def report(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input", "-i", help="Path to a previously generated report.json.", exists=True
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory for regenerated reports.")
    ] = Path("output_regenerated"),
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Comma-separated output formats: json, html, csv, console. Default: all.",
        ),
    ] = "html,csv,console",
) -> None:
    """Regenerate HTML, CSV, and console reports from an existing JSON report."""
    console.print(f"[dim]Loading report from {input_file}[/dim]")
    with open(input_file) as f:
        data = json.load(f)

    if "report" not in data:
        console.print(
            "[bold red]Invalid report format. Expected a top-level 'report' key.[/bold red]"
        )
        raise typer.Exit(1)

    formats = {f.strip().lower() for f in format.split(",")}
    eval_report = EvalReport(**data["report"])
    gen = ReportGenerator(eval_report)

    output_dir.mkdir(parents=True, exist_ok=True)

    if "console" in formats:
        gen.to_console()
    if "json" in formats:
        gen.to_json(output_dir / "report.json")
        console.print(f" ✓ JSON  → {output_dir / 'report.json'}")
    if "csv" in formats:
        gen.to_csv(output_dir / "report.csv")
        console.print(f" ✓ CSV   → {output_dir / 'report.csv'}")
    if "html" in formats:
        gen.to_html(output_dir / "report.html")
        console.print(f" ✓ HTML  → {output_dir / 'report.html'}")

    console.print(f"[bold green]✓ Reports regenerated in {output_dir}/[/bold green]")


@app.command(name="metrics")
def list_metrics() -> None:
    """List all available evaluation metrics."""
    MetricRegistry.discover_metrics()

    console.print("[bold magenta]Available Metrics:[/bold magenta]\n")
    for name, cls in sorted(MetricRegistry._registry.items()):
        needs_llm = _needs_param(cls, "llm_client")
        needs_embed = _needs_param(cls, "embed_client")
        tags = []
        if needs_llm:
            tags.append("[yellow]LLM[/yellow]")
        if needs_embed:
            tags.append("[blue]Embedding[/blue]")
        if not tags:
            tags.append("[green]standalone[/green]")
        tag_str = ", ".join(tags)
        console.print(f"  [bold cyan]{name:<25}[/bold cyan] {cls.__name__:<22} ({tag_str})")


if __name__ == "__main__":
    app()
