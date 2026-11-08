"""Typed data contracts shared across application and adapter boundaries."""

from dataclasses import dataclass
from enum import StrEnum


class RetrievalMode(StrEnum):
    """Supported retrieval modes; implementations arrive in later phases."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ExtractedFunction:
    """Source-preserving representation produced by a parser adapter."""

    symbol_key: str
    name: str
    qualified_name: str
    signature: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    embedding_input: str
    content_sha256: str
    docstring: str | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A ranked source result with an explicit score interpretation."""

    function_id: str
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str
    retrieval_mode: RetrievalMode
    rank: int
    score: float
    score_type: str


@dataclass(frozen=True, slots=True)
class IndexedFunction:
    """Persisted function identity paired with the data needed for embedding."""

    function_id: int
    function: ExtractedFunction


@dataclass(frozen=True, slots=True)
class IndexVersion:
    """Compatibility and artifact metadata for one semantic index build."""

    index_id: int
    repository_id: int
    model_id: str
    model_revision: str
    preprocessing_version: str
    embedding_dimension: int
    index_path: str
    status: str
    created_at: str
    artifact_sha256: str
    vector_count: int
    corpus_sha256: str


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Search results with model/index provenance when semantic retrieval ran."""

    results: tuple[SearchHit, ...]
    model_id: str | None
    index_id: int | None


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """FAISS-compatible integer vector ID and its raw similarity score."""

    vector_id: int
    score: float


@dataclass(frozen=True, slots=True)
class ConfiguredModel:
    """A model supported by configuration, independent of local weight state."""

    model_id: str
    role: str
    source: str
    weights_checked: bool = False


Embedding = tuple[float, ...]
