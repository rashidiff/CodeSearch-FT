"""Explicit interfaces at meaningful application and adapter boundaries."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from codesearch_ft.domain.models import (
    ConfiguredModel,
    Embedding,
    ExtractedFunction,
    IndexedFunction,
    IndexVersion,
    RetrievalMode,
    SearchHit,
    SearchOutcome,
    VectorMatch,
)


class CodeParser(Protocol):
    """Parses source text without importing or executing it."""

    def extract(self, source: str, relative_path: str) -> Sequence[ExtractedFunction]: ...


class CodeEncoder(Protocol):
    """Encodes query text and code into fixed-dimensional dense vectors."""

    model_id: str
    model_revision: str
    dimension: int

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Embedding]: ...

    def encode_code(self, snippets: Sequence[str]) -> Sequence[Embedding]: ...


class MetadataRepository(Protocol):
    """Persists repository, source, function, index, and vector metadata."""

    def get_or_create_repository(self, root_path: Path) -> int: ...

    def file_hash(self, repository_id: int, relative_path: str) -> str | None: ...

    def replace_file_functions(
        self,
        repository_id: int,
        relative_path: str,
        file_sha256: str,
        functions: Sequence[ExtractedFunction],
    ) -> None: ...

    def record_file_error(
        self, repository_id: int, relative_path: str, file_sha256: str
    ) -> None: ...

    def remove_source_file(self, repository_id: int, relative_path: str) -> None: ...

    def remove_unseen_files(self, repository_id: int, seen_paths: set[str]) -> int: ...

    def search_lexical(
        self, query: str, top_k: int, repository_id: int | None = None
    ) -> Sequence[SearchHit]: ...

    def list_indexed_functions(self, repository_id: int) -> Sequence[IndexedFunction]: ...

    def create_index_version(
        self,
        repository_id: int,
        model_id: str,
        model_revision: str,
        preprocessing_version: str,
        embedding_dimension: int,
        corpus_sha256: str,
        index_path: Path,
    ) -> IndexVersion: ...

    def fail_index_version(self, index_id: int) -> None: ...

    def add_vector_mappings(self, index_id: int, mappings: Sequence[tuple[int, int]]) -> None: ...

    def promote_index_version(
        self, index_id: int, artifact_sha256: str, vector_count: int
    ) -> IndexVersion: ...

    def get_index_version(self, index_id: int) -> IndexVersion | None: ...

    def list_index_versions(self, status: str | None = None) -> Sequence[IndexVersion]: ...

    def resolve_vector_matches(
        self, index_id: int, matches: Sequence[VectorMatch]
    ) -> Sequence[SearchHit]: ...

    def validate_vector_ids(self, index_id: int, vector_ids: Sequence[int]) -> None: ...


class VectorIndex(Protocol):
    """Stores and searches vectors by explicit integer IDs."""

    def add(self, vector_ids: Sequence[int], vectors: Sequence[Embedding]) -> None: ...

    def search(self, query: Embedding, top_k: int) -> Sequence[VectorMatch]: ...

    def save(self, destination: Path) -> None: ...

    def load(self, source: Path) -> None: ...


class LexicalRetriever(Protocol):
    """Returns source hits ordered by SQLite FTS5 BM25 rank."""

    def search(
        self, query: str, top_k: int, repository_id: int | None = None
    ) -> Sequence[SearchHit]: ...


class SemanticRetriever(Protocol):
    """Returns source hits ordered by normalized vector similarity."""

    def search(
        self, query: str, top_k: int, index_id: int | None = None
    ) -> Sequence[SearchHit]: ...


class SearchService(Protocol):
    """Coordinates one or more retrieval modes for a query."""

    def search(
        self,
        query: str,
        mode: RetrievalMode,
        top_k: int,
        index_id: int | None = None,
        repository_id: int | None = None,
    ) -> SearchOutcome: ...

    def list_indexes(self) -> Sequence[IndexVersion]: ...


class ExperimentTracker(Protocol):
    """Records configuration, metrics, and local experiment artifacts."""

    def log_params(self, values: dict[str, str | int | float | bool | None]) -> None: ...

    def log_metrics(self, values: dict[str, float], step: int | None = None) -> None: ...

    def log_artifact(self, path: Path) -> None: ...


class ModelCatalog(Protocol):
    """Describes supported models without forcing a model download."""

    def list_models(self) -> Sequence[ConfiguredModel]: ...
