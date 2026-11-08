"""Semantic retrieval over a validated FAISS index version."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock

from codesearch_ft.domain.errors import IndexCompatibilityError
from codesearch_ft.domain.models import IndexVersion, SearchHit
from codesearch_ft.domain.ports import CodeEncoder, MetadataRepository
from codesearch_ft.ml.semantic_index import SemanticIndexManager
from codesearch_ft.retrieval.faiss_index import FaissFlatIPIndex


class FaissSemanticRetriever:
    """Bind one encoder to compatible, current persisted index versions."""

    def __init__(
        self,
        repository: MetadataRepository,
        index_manager: SemanticIndexManager,
        encoder: CodeEncoder,
        preprocessing_version: str,
    ) -> None:
        self._repository = repository
        self._index_manager = index_manager
        self._encoder = encoder
        self._preprocessing_version = preprocessing_version
        self._loaded: dict[int, FaissFlatIPIndex] = {}
        self._lock = RLock()

    def search(self, query: str, top_k: int, index_id: int | None = None) -> Sequence[SearchHit]:
        version = _select_ready_version(self._repository, index_id)
        _ensure_compatible(version, self._encoder, self._preprocessing_version)
        with self._lock:
            index = self._loaded.get(version.index_id)
            if index is None:
                index = self._index_manager.load(version)
                self._loaded[version.index_id] = index
        embeddings = self._encoder.encode_queries((query,))
        if len(embeddings) != 1:
            raise IndexCompatibilityError("Encoder did not return exactly one query embedding.")
        vector_matches = index.search(embeddings[0], top_k)
        return self._repository.resolve_vector_matches(version.index_id, vector_matches)


def _select_ready_version(repository: MetadataRepository, index_id: int | None) -> IndexVersion:
    if index_id is not None:
        version = repository.get_index_version(index_id)
        if version is None:
            raise IndexCompatibilityError(f"Semantic index {index_id} does not exist.")
        if version.status != "ready":
            raise IndexCompatibilityError(
                f"Semantic index {index_id} is {version.status}; rebuild it before searching."
            )
        return version
    active = tuple(repository.list_index_versions(status="ready"))
    if not active:
        raise IndexCompatibilityError(
            "No active semantic index exists. Build one with `codesearch index --semantic`."
        )
    if len(active) > 1:
        raise IndexCompatibilityError(
            "More than one semantic index is active; pass --index-id to select one."
        )
    if not active:
        raise IndexCompatibilityError("No active semantic index is available.")
    return active[0]


def _ensure_compatible(
    version: IndexVersion, encoder: CodeEncoder, preprocessing_version: str
) -> None:
    actual = (
        version.model_id,
        version.model_revision,
        version.embedding_dimension,
        version.preprocessing_version,
    )
    expected = (
        encoder.model_id,
        encoder.model_revision,
        encoder.dimension,
        preprocessing_version,
    )
    if actual != expected:
        raise IndexCompatibilityError(
            f"Semantic index {version.index_id} is incompatible with the loaded encoder or "
            "preprocessing. Rebuild with the selected model revision."
        )
