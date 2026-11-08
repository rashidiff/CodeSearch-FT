"""Resolve the local metadata database location without touching repositories."""

import os
from pathlib import Path

import typer


def resolve_database_path(explicit_path: Path | None = None) -> Path:
    """Use an explicit path, environment override, then platform user data."""
    selected = explicit_path or (
        Path(os.environ["CODESEARCH_DATABASE"]) if os.environ.get("CODESEARCH_DATABASE") else None
    )
    if selected is not None:
        return selected.expanduser()
    return Path(typer.get_app_dir("codesearch-ft")) / "metadata.sqlite3"
