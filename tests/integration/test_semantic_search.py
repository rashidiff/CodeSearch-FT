"""Offline end-to-end semantic indexing, persistence and hybrid retrieval."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from codesearch_ft.application.indexing import RepositoryIndexingService
from codesearch_ft.application.search_service import LocalSearchService
from codesearch_ft.domain.errors import IndexCompatibilityError
from codesearch_ft.domain.models import Embedding, RetrievalMode
from codesearch_ft.ingestion.ast_parser import AstCodeParser
from codesearch_ft.ingestion.repository_scanner import RepositoryScanner
from codesearch_ft.ml.codebert_encoder import PREPROCESSING_VERSION
from codesearch_ft.ml.semantic_index import SemanticIndexManager
from codesearch_ft.storage.sqlite_repository import SQLiteRepository


class KeywordEncoder:
    model_id = "test/code-encoder"
    model_revision = "fixture-revision"
    dimension = 3

    def encode_queries(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple(_vector(text) for text in texts)

    def encode_code(self, snippets: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple(_vector(text) for text in snippets)


def _vector(text: str) -> Embedding:
    lowered = text.lower()
    if "red" in lowered:
        return (1.0, 0.0, 0.0)
    if "blue" in lowered:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _make_project(root: Path) -> None:
    root.mkdir()
    (root / "colors.py").write_text(
        'def red_item(): return "red"\ndef blue_item(): return "blue"\n',
        encoding="utf-8",
    )


def test_build_persist_reload_semantic_and_hybrid_results(tmp_path: Path) -> None:
    root = tmp_path / "project"
    _make_project(root)
    database = tmp_path / "metadata.sqlite3"
    artifact_root = tmp_path / "semantic_indexes"
    encoder = KeywordEncoder()

    with SQLiteRepository(database) as repository:
        summary = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository).index(
            root
        )
        manager = SemanticIndexManager(repository, artifact_root, batch_size=1)
        version = manager.rebuild(
            summary.repository_id, encoder, preprocessing_version=PREPROCESSING_VERSION
        )
        assert version.status == "ready"
        assert version.vector_count == 2
        assert Path(version.index_path).exists()
        assert Path(f"{version.index_path}.manifest.json").exists()
        loaded = manager.load(version)
        assert loaded.count == 2

        service = LocalSearchService(
            repository,
            artifact_root,
            encoder_provider=lambda: encoder,
        )
        semantic = service.search("red", RetrievalMode.SEMANTIC, 2, version.index_id)
        assert semantic.results[0].name == "red_item"
        assert abs(semantic.results[0].score - 1.0) < 1e-6
        assert semantic.results[0].score_type == "cosine_similarity"
        assert semantic.model_id == encoder.model_id

        hybrid = service.search("red", RetrievalMode.HYBRID, 2, version.index_id)
        assert hybrid.results[0].name == "red_item"
        assert hybrid.results[0].retrieval_mode is RetrievalMode.HYBRID
        assert hybrid.results[0].score_type == "reciprocal_rank_fusion(k=60)"

    with SQLiteRepository(database) as reopened:
        reloaded = SemanticIndexManager(reopened, artifact_root).load(version)
        assert reloaded.search((1.0, 0.0, 0.0), 2)[0].vector_id > 0


def test_source_updates_stale_active_index_and_rebuild_removes_deleted_vectors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _make_project(root)
    database = tmp_path / "metadata.sqlite3"
    artifact_root = tmp_path / "semantic_indexes"
    encoder = KeywordEncoder()
    with SQLiteRepository(database) as repository:
        service = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository)
        summary = service.index(root)
        manager = SemanticIndexManager(repository, artifact_root)
        original = manager.rebuild(
            summary.repository_id, encoder, preprocessing_version=PREPROCESSING_VERSION
        )

        (root / "colors.py").write_text('def green_item(): return "green"\n', encoding="utf-8")
        update = service.index(root, update=True)
        assert update.indexed_files == 1
        assert repository.get_index_version(original.index_id).status == "stale"  # type: ignore[union-attr]
        local_search = LocalSearchService(
            repository, artifact_root, encoder_provider=lambda: encoder
        )
        with pytest.raises(IndexCompatibilityError, match="stale"):
            local_search.search("blue", RetrievalMode.SEMANTIC, 5, original.index_id)

        replacement = manager.rebuild(
            summary.repository_id, encoder, preprocessing_version=PREPROCESSING_VERSION
        )
        assert replacement.status == "ready"
        assert replacement.vector_count == 1
        assert [
            hit.name for hit in local_search.search("green", RetrievalMode.SEMANTIC, 5).results
        ] == ["green_item"]
        with pytest.raises(IndexCompatibilityError, match="stale"):
            manager.load(original)


def test_failed_rebuild_preserves_previous_ready_index(tmp_path: Path) -> None:
    root = tmp_path / "project"
    _make_project(root)
    database = tmp_path / "metadata.sqlite3"
    artifacts = tmp_path / "semantic_indexes"
    with SQLiteRepository(database) as repository:
        summary = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository).index(
            root
        )
        manager = SemanticIndexManager(repository, artifacts)
        active = manager.rebuild(
            summary.repository_id,
            KeywordEncoder(),
            preprocessing_version=PREPROCESSING_VERSION,
        )

        class FailingEncoder(KeywordEncoder):
            def encode_code(self, snippets: Sequence[str]) -> tuple[Embedding, ...]:
                raise RuntimeError("fixture encoder failure")

        with pytest.raises(RuntimeError, match="fixture encoder failure"):
            manager.rebuild(
                summary.repository_id,
                FailingEncoder(),
                preprocessing_version=PREPROCESSING_VERSION,
            )
        assert repository.get_index_version(active.index_id).status == "ready"  # type: ignore[union-attr]
        assert manager.load(active).count == 2
        assert repository.list_index_versions(status="failed")


def test_model_revision_mismatch_and_artifact_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _make_project(root)
    database = tmp_path / "metadata.sqlite3"
    artifacts = tmp_path / "semantic_indexes"
    with SQLiteRepository(database) as repository:
        summary = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository).index(
            root
        )
        version = SemanticIndexManager(repository, artifacts).rebuild(
            summary.repository_id,
            KeywordEncoder(),
            preprocessing_version=PREPROCESSING_VERSION,
        )
        wrong_revision = KeywordEncoder()
        wrong_revision.model_revision = "different-revision"
        service = LocalSearchService(repository, artifacts, encoder_provider=lambda: wrong_revision)
        with pytest.raises(IndexCompatibilityError, match="incompatible"):
            service.search("red", RetrievalMode.SEMANTIC, 5, version.index_id)

        Path(version.index_path).write_bytes(b"tampered")
        with pytest.raises(IndexCompatibilityError, match="SHA-256"):
            SemanticIndexManager(repository, artifacts).load(version)
