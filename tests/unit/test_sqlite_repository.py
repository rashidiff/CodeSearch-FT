"""Migration, transaction, FTS and BM25 integration fixtures."""

import sqlite3
from pathlib import Path

import pytest

from codesearch_ft.domain.errors import StorageError
from codesearch_ft.domain.models import ExtractedFunction
from codesearch_ft.storage import migrations
from codesearch_ft.storage.migrations import migrate
from codesearch_ft.storage.sqlite_repository import SQLiteRepository


def _function(name: str, code: str, symbol_key: str | None = None) -> ExtractedFunction:
    return ExtractedFunction(
        symbol_key=symbol_key or name,
        name=name,
        qualified_name=name,
        signature=f"def {name}()",
        file_path="sample.py",
        line_start=1,
        line_end=2,
        source_code=code,
        embedding_input=code,
        content_sha256=f"hash-{name}",
    )


def test_migrations_are_numbered_idempotent_and_enforce_foreign_keys(tmp_path: Path) -> None:
    database = tmp_path / "schema.sqlite3"
    connection = sqlite3.connect(database)
    migrate(connection)
    migrate(connection)

    assert connection.execute("SELECT version FROM schema_migrations").fetchall() == [
        (1,),
        (2,),
    ]
    assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO source_files(repository_id, relative_path, sha256, indexed_at, status) "
            "VALUES (999, 'x.py', 'h', 'now', 'indexed')"
        )
    connection.close()


def test_failed_migration_rolls_back_ddl_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_create(connection: sqlite3.Connection) -> None:
        connection.executescript(
            "BEGIN IMMEDIATE; CREATE TABLE partial_migration(id INTEGER); "
            "CREATE TABLE invalid syntax;"
        )

    monkeypatch.setattr(migrations, "MIGRATIONS", ((99, fail_after_create),))
    connection = sqlite3.connect(tmp_path / "rollback.sqlite3")

    with pytest.raises(StorageError):
        migrate(connection)

    assert (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='partial_migration'"
        ).fetchone()
        is None
    )
    assert connection.execute("SELECT version FROM schema_migrations").fetchall() == []
    connection.close()


def test_migration_runner_applies_new_numbered_migration_to_existing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "upgrade.sqlite3"
    connection = sqlite3.connect(database)
    migrate(connection)

    def migration_2(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE migration_two_marker(id INTEGER PRIMARY KEY)")

    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (*migrations.MIGRATIONS, (3, migration_2)),
    )
    migrate(connection)

    assert connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall() == [
        (1,),
        (2,),
        (3,),
    ]
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_two_marker'"
    ).fetchone() == ("migration_two_marker",)
    connection.close()


def test_fts_synchronizes_insert_update_and_delete_and_ranks_bm25_ascending(
    tmp_path: Path,
) -> None:
    database = tmp_path / "search.sqlite3"
    with SQLiteRepository(database) as repository:
        repo_id = repository.get_or_create_repository(tmp_path)
        repository.replace_file_functions(
            repo_id,
            "sample.py",
            "file-hash",
            (
                _function(
                    "remove_duplicates",
                    "def remove_duplicates():\n    return list(dict.fromkeys(items))",
                ),
                _function("deduplicate", "def deduplicate():\n    return sorted(set(items))"),
            ),
        )
        hits = repository.search_lexical("preserve original order", 10)
        assert len(hits) == 0  # No matching terms exist in indexed source text.
        hits = repository.search_lexical("dict fromkeys", 10)
        assert hits[0].qualified_name == "remove_duplicates"
        assert hits[0].score <= hits[-1].score
        assert hits[0].score_type == "sqlite_fts5_bm25 (lower is better)"
        assert hits[0].file_path == str(tmp_path / "sample.py")
        assert hits[0].line_start == 1
        assert "dict.fromkeys" in hits[0].code_snippet

        repository.replace_file_functions(
            repo_id,
            "sample.py",
            "new-hash",
            (
                _function(
                    "rename_items",
                    "def rename_items():\n    return list(dict.fromkeys(items))",
                    "remove_duplicates",
                ),
            ),
        )
        assert [hit.name for hit in repository.search_lexical("rename", 10)] == ["rename_items"]
        assert repository.search_lexical("deduplicate", 10) == ()
        repository.remove_source_file(repo_id, "sample.py")
        assert repository.search_lexical("rename", 10) == ()


def test_bm25_ties_use_stable_function_id_order(tmp_path: Path) -> None:
    with SQLiteRepository(tmp_path / "search.sqlite3") as repository:
        repo_id = repository.get_or_create_repository(tmp_path)
        repository.replace_file_functions(
            repo_id,
            "x.py",
            "x",
            (
                _function("first", "def sharedtoken(): return 1"),
                _function("second", "def sharedtoken(): return 1"),
            ),
        )

        hits = repository.search_lexical("sharedtoken", 10)

        assert [hit.name for hit in hits] == ["first", "second"]
        assert hits[0].score == hits[1].score


def test_lexical_parser_handles_empty_operator_and_injection_like_queries(tmp_path: Path) -> None:
    with SQLiteRepository(tmp_path / "search.sqlite3") as repository:
        repo_id = repository.get_or_create_repository(tmp_path)
        repository.replace_file_functions(
            repo_id, "x.py", "x", (_function("select", "def select():\n    return 'safe'"),)
        )
        assert repository.search_lexical("   *** AND OR", 5) == ()
        assert repository.search_lexical('" OR * MATCH', 5) == ()
        assert len(repository.search_lexical("select", 5)) == 1


def test_failed_file_update_rolls_back_prior_fts_and_metadata(tmp_path: Path) -> None:
    with SQLiteRepository(tmp_path / "search.sqlite3") as repository:
        repo_id = repository.get_or_create_repository(tmp_path)
        repository.replace_file_functions(
            repo_id, "x.py", "x", (_function("old", "def old(): pass"),)
        )
        connection = sqlite3.connect(repository.database_path)
        connection.execute(
            "CREATE TRIGGER force_update_failure BEFORE UPDATE ON functions "
            "BEGIN SELECT RAISE(ABORT, 'fixture failure'); END"
        )
        connection.commit()
        connection.close()
        with pytest.raises(StorageError):
            repository.replace_file_functions(
                repo_id,
                "x.py",
                "bad",
                (_function("old", "def changed(): pass"),),
            )
        assert [hit.name for hit in repository.search_lexical("old", 5)] == ["old"]
