"""Versioned API contract tests with local synthetic vectors only."""

from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from codesearch_ft.api.app import create_app
from codesearch_ft.application.indexing import RepositoryIndexingService
from codesearch_ft.application.search_service import LocalSearchService
from codesearch_ft.domain.models import Embedding
from codesearch_ft.ingestion.ast_parser import AstCodeParser
from codesearch_ft.ingestion.repository_scanner import RepositoryScanner
from codesearch_ft.ml.codebert_encoder import PREPROCESSING_VERSION
from codesearch_ft.ml.semantic_index import SemanticIndexManager
from codesearch_ft.storage.sqlite_repository import SQLiteRepository


class KeywordEncoder:
    model_id = "test/api-encoder"
    model_revision = "api-fixture-v1"
    dimension = 2

    def encode_queries(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple(_embedding(text) for text in texts)

    def encode_code(self, snippets: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple(_embedding(text) for text in snippets)


def _embedding(text: str) -> Embedding:
    return (1.0, 0.0) if "alpha" in text.lower() else (0.0, 1.0)


def test_api_models_indexes_and_all_search_modes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "functions.py").write_text(
        'def alpha_function(): return "alpha"\ndef beta_function(): return "beta"\n',
        encoding="utf-8",
    )
    database = tmp_path / "metadata.sqlite3"
    artifacts = tmp_path / "semantic_indexes"
    encoder = KeywordEncoder()
    repository = SQLiteRepository(database)
    indexed = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository).index(
        root
    )
    version = SemanticIndexManager(repository, artifacts).rebuild(
        indexed.repository_id,
        encoder,
        preprocessing_version=PREPROCESSING_VERSION,
    )
    service = LocalSearchService(repository, artifacts, encoder_provider=lambda: encoder)

    with TestClient(create_app(service)) as client:
        health = client.get("/health")
        models = client.get("/v1/models")
        indexes = client.get("/v1/indexes")
        lexical = client.post("/v1/search", json={"query": "alpha", "mode": "lexical"})
        semantic = client.post(
            "/v1/search",
            json={"query": "alpha", "mode": "semantic", "index_id": version.index_id},
        )
        hybrid = client.post(
            "/v1/search",
            json={"query": "alpha", "mode": "hybrid", "index_id": version.index_id},
        )
        openapi = client.get("/openapi.json")

    assert health.status_code == 200 and health.json()["search_ready"] is True
    assert models.status_code == 200
    assert models.json()[0]["weights_checked"] is False
    assert indexes.status_code == 200
    assert indexes.json()[0]["index_id"] == version.index_id
    assert "index_path" not in indexes.json()[0]
    assert lexical.status_code == 200
    assert lexical.json()["results"][0]["name"] == "alpha_function"
    assert lexical.json()["results"][0]["score_type"].startswith("sqlite_fts5_bm25")
    assert semantic.status_code == 200
    assert semantic.json()["model_id"] == encoder.model_id
    assert semantic.json()["results"][0]["name"] == "alpha_function"
    assert semantic.json()["results"][0]["score_type"] == "cosine_similarity"
    assert hybrid.status_code == 200
    assert hybrid.json()["results"][0]["score_type"].startswith("reciprocal_rank_fusion")
    assert "/v1/search" in openapi.json()["paths"]
    repository.close()


def test_api_validation_unknown_index_and_unavailable_search_service(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "metadata.sqlite3"
    with SQLiteRepository(database) as repository:
        service = LocalSearchService(repository, tmp_path / "artifacts")
        with TestClient(create_app(service)) as client:
            empty = client.post("/v1/search", json={"query": ""})
            too_long = client.post("/v1/search", json={"query": "x" * 2_001})
            bad_limit = client.post("/v1/search", json={"query": "x", "top_k": 101})
            extra = client.post("/v1/search", json={"query": "x", "path": "C:/"})
            unknown = client.post(
                "/v1/search", json={"query": "x", "mode": "semantic", "index_id": 999}
            )
        with TestClient(create_app()) as unconfigured:
            unavailable = unconfigured.post("/v1/search", json={"query": "x"})

    for response in (empty, too_long, bad_limit, extra):
        assert response.status_code == 422
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "index_not_found"
    assert unavailable.status_code == 503
    assert "/v1/indexing" not in create_app().openapi()["paths"]
