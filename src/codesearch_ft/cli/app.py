"""Typer entry point for real foundation-stage local operations."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import typer
import uvicorn

from codesearch_ft.api.app import create_app
from codesearch_ft.application.database_path import resolve_database_path
from codesearch_ft.application.indexing import RepositoryIndexingService
from codesearch_ft.application.model_catalog import list_configured_models
from codesearch_ft.application.search_service import LocalSearchService
from codesearch_ft.config.loader import load_evaluation_config, load_training_config
from codesearch_ft.domain.errors import CodeSearchError
from codesearch_ft.domain.models import IndexVersion, RetrievalMode
from codesearch_ft.ingestion.ast_parser import AstCodeParser
from codesearch_ft.ingestion.repository_scanner import IndexingOptions, RepositoryScanner
from codesearch_ft.ml.codebert_encoder import PREPROCESSING_VERSION, CodeBERTEncoder
from codesearch_ft.ml.semantic_index import SemanticIndexManager
from codesearch_ft.storage.sqlite_repository import SQLiteRepository
from codesearch_ft.structured_logging import configure_logging

app = typer.Typer(
    name="codesearch",
    help="Local semantic search for Python source code.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
models_app = typer.Typer(help="Inspect supported model identifiers.", no_args_is_help=True)
config_app = typer.Typer(help="Validate local YAML configuration.", no_args_is_help=True)
indexes_app = typer.Typer(help="Inspect persisted semantic index versions.", no_args_is_help=True)
train_app = typer.Typer(help="Run explicit configured fine-tuning jobs.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")
app.add_typer(indexes_app, name="indexes")


@app.command("train")
def train_model(
    config_path: Annotated[Path, typer.Option("--config", help="Training profile YAML.")],
    resume: Annotated[
        Path | None, typer.Option("--resume", help="Resume an epoch checkpoint.")
    ] = None,
) -> None:
    """Run an explicitly selected local training profile."""
    try:
        from codesearch_ft.ml.trainer import train

        result = train(load_training_config(config_path), resume=resume)
    except (CodeSearchError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command("benchmark")
def run_evaluation(
    config_path: Annotated[Path, typer.Option("--config", help="Fixed-pool evaluation YAML.")],
) -> None:
    """Compare BM25 and pretrained/fine-tuned CodeBERT on fixed held-out candidates."""
    try:
        from codesearch_ft.evaluation.benchmark import run_benchmark

        config = load_evaluation_config(config_path)
        result = run_benchmark(config)
    except (CodeSearchError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "results": result["results"],
                "dataset": result["dataset"],
                "report": str(config.output_dir),
            },
            indent=2,
        )
    )


@app.command("index")
def index_repository(
    root: Annotated[Path, typer.Argument(help="Local Python repository directory.")],
    update: bool = typer.Option(
        False, "--update", help="Skip unchanged files and reconcile stale files."
    ),
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite path; defaults to user data directory."),
    ] = None,
    max_file_bytes: int = typer.Option(1_000_000, min=1, help="Maximum source file size in bytes."),
    max_files: int = typer.Option(100_000, min=1, help="Maximum Python files scanned."),
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Glob pattern to exclude (repeatable)."),
    ] = None,
    semantic: bool = typer.Option(
        False,
        "--semantic",
        help="Also build a versioned semantic index using locally cached CodeBERT weights.",
    ),
) -> None:
    """Index Python function metadata locally without executing repository code."""
    try:
        options = IndexingOptions(
            max_file_bytes=max_file_bytes,
            max_files=max_files,
            excluded_patterns=(*IndexingOptions().excluded_patterns, *(exclude or [])),
        )
        db_path = resolve_database_path(database)
        with SQLiteRepository(db_path) as repository:
            service = RepositoryIndexingService(
                RepositoryScanner(options), AstCodeParser(), repository
            )
            summary = service.index(root, update=update)
            semantic_version: IndexVersion | None = None
            if semantic:
                encoder = CodeBERTEncoder.from_pretrained(local_files_only=True)
                semantic_version = SemanticIndexManager(
                    repository, db_path.parent / "semantic_indexes"
                ).rebuild(
                    summary.repository_id,
                    encoder,
                    preprocessing_version=PREPROCESSING_VERSION,
                )
            stale_versions = tuple(repository.list_index_versions(status="stale"))
    except (CodeSearchError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Indexed {summary.indexed_files} files ({summary.function_count} functions); "
        f"unchanged={summary.unchanged_files}, removed={summary.removed_files}, "
        f"skipped={len(summary.issues)}. Database: {db_path}"
    )
    for issue in summary.issues:
        typer.echo(f"Skipped {issue.relative_path}: {issue.reason}", err=True)
    if semantic_version is not None:
        typer.echo(
            f"Semantic index ready: id={semantic_version.index_id}, "
            f"model={semantic_version.model_id}@{semantic_version.model_revision}, "
            f"vectors={semantic_version.vector_count}"
        )
    elif stale_versions:
        typer.echo(
            "A previous semantic index is stale after source changes. "
            "Rebuild with `codesearch index <repository> --update --semantic`.",
            err=True,
        )


@app.command("search")
def search_lexical(
    query: Annotated[str, typer.Argument(help="Natural language or code terms to search.")],
    mode: Annotated[RetrievalMode, typer.Option(help="Retrieval mode.")] = RetrievalMode.LEXICAL,
    top_k: int = typer.Option(10, min=1, max=1000, help="Maximum results."),
    rrf_constant: int = typer.Option(60, min=1, help="Positive RRF rank-fusion constant."),
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite path; defaults to user data directory."),
    ] = None,
    index_id: int | None = typer.Option(None, min=1, help="Semantic index version ID."),
) -> None:
    """Search indexed functions lexically, semantically, or with rank fusion."""
    db_path = resolve_database_path(database)
    if not db_path.exists():
        typer.echo(
            f"Index database not found at {db_path}. Run `codesearch index <repository>` first "
            "or pass --database / set CODESEARCH_DATABASE.",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        with SQLiteRepository(db_path) as repository:
            outcome = LocalSearchService(
                repository,
                db_path.parent / "semantic_indexes",
                local_files_only=True,
                rrf_constant=rrf_constant,
            ).search(query, mode, top_k, index_id)
    except (CodeSearchError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not outcome.results:
        typer.echo(f"No {mode.value} matches.")
        return
    if outcome.model_id:
        typer.echo(f"Model: {outcome.model_id}; index: {outcome.index_id}")
    for hit in outcome.results:
        score = (
            f"BM25={hit.score:.6f}"
            if hit.score_type.startswith("sqlite_fts5_bm25")
            else f"{hit.score_type}={hit.score:.6f}"
        )
        typer.echo(
            f"{hit.rank}. {hit.qualified_name} "
            f"({hit.file_path}:{hit.line_start}-{hit.line_end}) {score}"
        )
        typer.echo(hit.code_snippet.rstrip())


@app.callback()
def configure_app(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = typer.Option(
        "INFO", "--log-level", help="Structured application log level."
    ),
) -> None:
    """Configure standard-library JSON logging for the command invocation."""
    configure_logging(log_level)


@models_app.command("list")
def models_list(
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """List supported model identifiers without loading weights or using the network."""
    models = list_configured_models()
    if as_json:
        typer.echo(json.dumps([asdict(model) for model in models], indent=2))
        return

    typer.echo("Configured models (catalog only; weights are not checked or downloaded):")
    for model in models:
        typer.echo(f"- {model.model_id} - {model.role} ({model.source})")


@models_app.command("download")
def models_download() -> None:
    """Explicitly download/load pinned CodeBERT weights into the local HF cache."""
    try:
        encoder = CodeBERTEncoder.from_pretrained(local_files_only=False)
    except (CodeSearchError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Cached {encoder.model_id}@{encoder.model_revision} "
        f"({encoder.dimension} dimensions) in the Hugging Face cache."
    )


@models_app.command("smoke")
def models_smoke() -> None:
    """Run one real CPU query/code embedding using weights already in the local cache."""
    try:
        encoder = CodeBERTEncoder.from_pretrained(local_files_only=True)
        query = encoder.encode_queries(("remove duplicate items while preserving order",))
        code = encoder.encode_code(("def unique(items):\n    return list(dict.fromkeys(items))",))
    except (CodeSearchError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if len(query) != 1 or len(code) != 1:
        typer.echo("Error: CodeBERT smoke inference returned an unexpected batch size.", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"CPU inference succeeded for {encoder.model_id}@{encoder.model_revision}: "
        f"dimension={encoder.dimension}, query_vectors={len(query)}, code_vectors={len(code)}."
    )


@indexes_app.command("list")
def indexes_list(
    database: Annotated[
        Path | None,
        typer.Option("--database", help="SQLite path; defaults to user data directory."),
    ] = None,
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """List ready, stale, building, and failed semantic index versions."""
    db_path = resolve_database_path(database)
    if not db_path.exists():
        typer.echo("No index database exists yet. Run `codesearch index <repository>` first.")
        return
    with SQLiteRepository(db_path) as repository:
        versions = repository.list_index_versions()
    entries = [
        {
            "index_id": item.index_id,
            "repository_id": item.repository_id,
            "model_id": item.model_id,
            "model_revision": item.model_revision,
            "embedding_dimension": item.embedding_dimension,
            "vector_count": item.vector_count,
            "status": item.status,
            "created_at": item.created_at,
        }
        for item in versions
    ]
    if as_json:
        typer.echo(json.dumps(entries, indent=2))
        return
    if not entries:
        typer.echo("No semantic indexes have been built.")
        return
    for entry in entries:
        typer.echo(
            f"{entry['index_id']}: {entry['status']} - {entry['model_id']}@"
            f"{entry['model_revision']} ({entry['vector_count']} vectors, "
            f"{entry['embedding_dimension']} dimensions)"
        )


@config_app.command("validate")
def config_validate(
    path: Annotated[Path, typer.Argument(help="YAML configuration path.")],
    kind: Literal["training", "evaluation"] = typer.Option(
        "training", "--kind", help="Configuration schema to validate."
    ),
) -> None:
    """Validate a training or evaluation profile; this command does not execute it."""
    try:
        config = load_training_config(path) if kind == "training" else load_evaluation_config(path)
    except CodeSearchError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    profile = config.profile
    typer.echo(f"Configuration is valid: kind={kind}, profile={profile}. No work was started.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (defaults to localhost)."),
    port: int = typer.Option(8000, min=1, max=65535, help="HTTP port."),
) -> None:
    """Run the local API; model weights load only for semantic/hybrid requests."""
    db_path = resolve_database_path()
    repository = SQLiteRepository(db_path)
    service = LocalSearchService(
        repository,
        db_path.parent / "semantic_indexes",
        local_files_only=True,
    )
    try:
        uvicorn.run(create_app(service), host=host, port=port, log_config=None)
    finally:
        repository.close()


def main() -> None:
    """Console-script callable."""
    app()
