"""Local search composition for lexical, semantic, and hybrid modes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from threading import RLock

from codesearch_ft.domain.errors import IndexCompatibilityError, IndexNotFoundError
from codesearch_ft.domain.models import IndexVersion, RetrievalMode, SearchOutcome
from codesearch_ft.domain.ports import CodeEncoder, MetadataRepository
from codesearch_ft.ml.codebert_encoder import PREPROCESSING_VERSION, CodeBERTEncoder
from codesearch_ft.ml.semantic_index import SemanticIndexManager
from codesearch_ft.retrieval.hybrid import reciprocal_rank_fusion
from codesearch_ft.retrieval.lexical import FTS5LexicalRetriever
from codesearch_ft.retrieval.semantic import FaissSemanticRetriever


class LocalSearchService:
    """Compose real retrieval adapters and lazily load local model weights."""

    def __init__(
        self,
        repository: MetadataRepository,
        artifact_root: Path,
        *,
        encoder_provider: Callable[[], CodeEncoder] | None = None,
        local_files_only: bool = True,
        rrf_constant: int = 60,
        preprocessing_version: str = PREPROCESSING_VERSION,
    ) -> None:
        if rrf_constant < 1:
            raise ValueError("RRF constant must be positive.")
        self._repository = repository
        self._artifact_root = artifact_root
        self._encoder_provider = encoder_provider or (
            lambda: CodeBERTEncoder.from_pretrained(local_files_only=local_files_only)
        )
        self._rrf_constant = rrf_constant
        self._preprocessing_version = preprocessing_version
        self._encoder: CodeEncoder | None = None
        self._semantic: FaissSemanticRetriever | None = None
        self._lexical = FTS5LexicalRetriever(repository)
        self._initialization_lock = RLock()

    def search(
        self,
        query: str,
        mode: RetrievalMode,
        top_k: int,
        index_id: int | None = None,
        repository_id: int | None = None,
    ) -> SearchOutcome:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if len(query) > 2_000:
            raise ValueError("Search query exceeds the 2,000 character limit.")
        if not 1 <= top_k <= 1_000:
            raise ValueError("top_k must be between 1 and 1000.")

        if mode is RetrievalMode.LEXICAL:
            repository_id = None
            if index_id is not None:
                version = self._repository.get_index_version(index_id)
                if version is None:
                    raise IndexNotFoundError(f"Semantic index {index_id} does not exist.")
                if repository_id is not None and repository_id != version.repository_id:
                    raise IndexCompatibilityError(
                        "The selected index does not belong to the requested repository."
                    )
                repository_id = version.repository_id
            results = tuple(self._lexical.search(query, top_k, repository_id))
            return SearchOutcome(results=results, model_id=None, index_id=index_id)

        version = _select_active_index(self._repository, index_id)
        if repository_id is not None and repository_id != version.repository_id:
            raise IndexCompatibilityError(
                "The selected semantic index does not belong to the requested repository."
            )
        semantic_results = tuple(self._semantic_retriever().search(query, top_k, version.index_id))
        if mode is RetrievalMode.SEMANTIC:
            return SearchOutcome(
                results=semantic_results,
                model_id=version.model_id,
                index_id=version.index_id,
            )
        lexical_results = tuple(self._lexical.search(query, top_k, version.repository_id))
        results = reciprocal_rank_fusion(
            lexical_results,
            semantic_results,
            top_k=top_k,
            constant=self._rrf_constant,
        )
        return SearchOutcome(results, version.model_id, version.index_id)

    def list_indexes(self) -> Sequence[IndexVersion]:
        return self._repository.list_index_versions()

    def _semantic_retriever(self) -> FaissSemanticRetriever:
        with self._initialization_lock:
            if self._semantic is None:
                if self._encoder is None:
                    self._encoder = self._encoder_provider()
                manager = SemanticIndexManager(self._repository, self._artifact_root)
                self._semantic = FaissSemanticRetriever(
                    self._repository,
                    manager,
                    self._encoder,
                    self._preprocessing_version,
                )
            return self._semantic


def _select_active_index(repository: MetadataRepository, index_id: int | None) -> IndexVersion:
    if index_id is not None:
        version = repository.get_index_version(index_id)
        if version is None:
            raise IndexNotFoundError(f"Semantic index {index_id} does not exist.")
        if version.status != "ready":
            raise IndexCompatibilityError(
                f"Semantic index {index_id} is {version.status}; rebuild it before searching."
            )
        return version
    versions = tuple(repository.list_index_versions(status="ready"))
    if not versions:
        raise IndexCompatibilityError(
            "No active semantic index exists. Build one with `codesearch index --semantic`."
        )
    if len(versions) != 1:
        raise IndexCompatibilityError(
            "More than one semantic index is active; pass --index-id to select one."
        )
    return versions[0]
