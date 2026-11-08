"""Validated, versioned HTTP request and response models."""

from pydantic import BaseModel, ConfigDict, Field

from codesearch_ft.domain.models import RetrievalMode


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchFilters(APIModel):
    repository_id: int | None = Field(default=None, ge=1)


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=2_000)
    index_id: int | None = Field(default=None, ge=1)
    mode: RetrievalMode = RetrievalMode.LEXICAL
    top_k: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters | None = None


class SearchResultResponse(APIModel):
    function_id: str
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str
    rank: int
    score: float
    score_type: str


class SearchResponse(APIModel):
    query: str
    model_id: str | None
    index_id: int | None
    retrieval_mode: RetrievalMode
    elapsed_ms: float
    results: list[SearchResultResponse]


class ModelResponse(APIModel):
    model_id: str
    role: str
    source: str
    weights_checked: bool


class IndexResponse(APIModel):
    index_id: int
    repository_id: int
    model_id: str
    model_revision: str
    preprocessing_version: str
    embedding_dimension: int
    vector_count: int
    status: str
    created_at: str


class ErrorBody(APIModel):
    code: str
    message: str


class ErrorResponse(APIModel):
    error: ErrorBody
