"""Database location override precedence."""

from pathlib import Path

from pytest import MonkeyPatch

from codesearch_ft.application.database_path import resolve_database_path


def test_explicit_database_path_precedes_environment(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    environment_path = tmp_path / "environment.sqlite3"
    explicit_path = tmp_path / "explicit.sqlite3"
    monkeypatch.setenv("CODESEARCH_DATABASE", str(environment_path))

    assert resolve_database_path(explicit_path) == explicit_path


def test_database_path_uses_environment_override(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    environment_path = tmp_path / "environment.sqlite3"
    monkeypatch.setenv("CODESEARCH_DATABASE", str(environment_path))

    assert resolve_database_path() == environment_path
