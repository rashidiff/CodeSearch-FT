"""End-to-end indexing, persistence and lexical retrieval over a temp repository."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from codesearch_ft.application.indexing import RepositoryIndexingService
from codesearch_ft.cli.app import app
from codesearch_ft.domain.errors import IndexingError
from codesearch_ft.domain.models import Embedding
from codesearch_ft.domain.ports import CodeEncoder
from codesearch_ft.ingestion.ast_parser import AstCodeParser
from codesearch_ft.ingestion.repository_scanner import IndexingOptions, RepositoryScanner
from codesearch_ft.ml.codebert_encoder import CodeBERTEncoder
from codesearch_ft.retrieval.lexical import FTS5LexicalRetriever
from codesearch_ft.storage.sqlite_repository import SQLiteRepository


def test_index_and_update_skip_modified_deleted_and_renamed_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "alpha.py").write_text(
        "def remove_duplicates(items):\n"
        '    """Keep first occurrence order."""\n'
        "    return list(dict.fromkeys(items))\n",
        encoding="utf-8",
    )
    (root / "gone.py").write_text("def obsolete(): return 1\n", encoding="utf-8")
    database = tmp_path / "index.sqlite3"

    with SQLiteRepository(database) as repository:
        service = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository)
        first = service.index(root)
        assert first.indexed_files == 2
        assert first.function_count == 2
        second = service.index(root, update=True)
        assert second.unchanged_files == 2
        assert second.indexed_files == 0

    (root / "alpha.py").write_text(
        "def preserve_unique(items): return list(dict.fromkeys(items))\n", encoding="utf-8"
    )
    (root / "gone.py").unlink()
    (root / "renamed.py").write_text("def renamed(): return 'present'\n", encoding="utf-8")
    with SQLiteRepository(database) as repository:
        service = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository)
        third = service.index(root, update=True)
        assert third.indexed_files == 2
        assert third.removed_files == 1

    with SQLiteRepository(database) as reopened:
        assert [hit.name for hit in FTS5LexicalRetriever(reopened).search("obsolete", 5)] == []
        assert [hit.name for hit in FTS5LexicalRetriever(reopened).search("unique", 5)] == [
            "preserve_unique"
        ]
        assert [hit.name for hit in FTS5LexicalRetriever(reopened).search("present", 5)] == [
            "renamed"
        ]


def test_incomplete_scan_preserves_existing_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "one.py").write_text("def one(): pass\n", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    with SQLiteRepository(database) as repository:
        RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository).index(root)
    (root / "two.py").write_text("def two(): pass\n", encoding="utf-8")
    with SQLiteRepository(database) as repository:
        service = RepositoryIndexingService(
            RepositoryScanner(options=IndexingOptions(max_files=1)),
            AstCodeParser(),
            repository,
        )
        with pytest.raises(IndexingError):
            service.index(root, update=True)
        assert [hit.name for hit in FTS5LexicalRetriever(repository).search("one", 10)] == ["one"]


def test_syntax_error_is_reported_and_stale_functions_are_removed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source_file = root / "module.py"
    source_file.write_text("def working_function(): return 'searchable'\n", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    with SQLiteRepository(database) as repository:
        service = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository)
        service.index(root)
        assert FTS5LexicalRetriever(repository).search("working_function", 5)

    source_file.write_text("def broken_function(:\n", encoding="utf-8")
    with SQLiteRepository(database) as repository:
        summary = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository).index(
            root, update=True
        )
        assert summary.issues[0].reason == "Python syntax could not be parsed"
        assert FTS5LexicalRetriever(repository).search("working_function", 5) == ()


def test_cli_index_and_lexical_search_with_explicit_database(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "x.py").write_text("def read_json(path): return json.load(path)\n", encoding="utf-8")
    database = tmp_path / "custom.sqlite3"
    runner = CliRunner()

    indexed = runner.invoke(app, ["index", str(root), "--database", str(database)])
    searched = runner.invoke(app, ["search", "json", "--database", str(database)])

    assert indexed.exit_code == 0, indexed.output
    assert "1 files (1 functions)" in indexed.output
    assert searched.exit_code == 0, searched.output
    assert "read_json" in searched.output
    assert "BM25=" in searched.output


def test_cli_semantic_mode_requires_active_index_and_missing_database(tmp_path: Path) -> None:
    runner = CliRunner()
    existing_db = tmp_path / "empty.sqlite3"
    with SQLiteRepository(existing_db):
        pass
    semantic = runner.invoke(
        app,
        ["search", "query", "--mode", "semantic", "--database", str(existing_db)],
    )
    missing = runner.invoke(
        app, ["search", "query", "--database", str(tmp_path / "missing.sqlite3")]
    )

    assert semantic.exit_code == 1
    assert "No active semantic index" in semantic.output
    assert missing.exit_code == 1
    assert "Run `codesearch index" in missing.output


def test_cli_builds_and_searches_semantic_and_hybrid_with_offline_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixtureEncoder:
        model_id = "test/cli-encoder"
        model_revision = "fixture-revision"
        dimension = 2

        def encode_queries(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
            return tuple(_fixture_vector(text) for text in texts)

        def encode_code(self, snippets: Sequence[str]) -> tuple[Embedding, ...]:
            return tuple(_fixture_vector(text) for text in snippets)

    def fixture_encoder(**kwargs: Any) -> CodeEncoder:
        del kwargs
        return FixtureEncoder()

    monkeypatch.setattr(
        CodeBERTEncoder, "from_pretrained", cast(Any, staticmethod(fixture_encoder))
    )
    root = tmp_path / "repo"
    root.mkdir()
    (root / "x.py").write_text('def alpha_function(): return "alpha"\n', encoding="utf-8")
    database = tmp_path / "semantic.sqlite3"
    runner = CliRunner()

    indexed = runner.invoke(app, ["index", str(root), "--semantic", "--database", str(database)])
    semantic = runner.invoke(
        app,
        [
            "search",
            "alpha",
            "--mode",
            "semantic",
            "--database",
            str(database),
        ],
    )
    hybrid = runner.invoke(
        app,
        ["search", "alpha", "--mode", "hybrid", "--database", str(database)],
    )
    smoke = runner.invoke(app, ["models", "smoke"])
    indexes = runner.invoke(app, ["indexes", "list", "--database", str(database), "--json"])

    assert indexed.exit_code == 0, indexed.output
    assert "Semantic index ready" in indexed.output
    assert semantic.exit_code == 0, semantic.output
    assert "alpha_function" in semantic.output
    assert "cosine_similarity" in semantic.output
    assert hybrid.exit_code == 0, hybrid.output
    assert "reciprocal_rank_fusion" in hybrid.output
    assert smoke.exit_code == 0, smoke.output
    assert "CPU inference succeeded" in smoke.output
    assert indexes.exit_code == 0 and '"status": "ready"' in indexes.output


def _fixture_vector(text: str) -> Embedding:
    return (1.0, 0.0) if "alpha" in text.lower() else (0.0, 1.0)
