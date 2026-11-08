"""FastAPI application factory and versioned local search API."""

import logging
from time import perf_counter

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from codesearch_ft import __version__
from codesearch_ft.api.schemas import (
    APIModel,
    ErrorResponse,
    IndexResponse,
    ModelResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from codesearch_ft.application.model_catalog import list_configured_models
from codesearch_ft.domain.errors import (
    CodeSearchError,
    IndexCompatibilityError,
    IndexNotFoundError,
    ModelUnavailableError,
)
from codesearch_ft.domain.ports import SearchService

logger = logging.getLogger(__name__)


class HealthResponse(APIModel):
    status: str
    service: str
    version: str
    search_ready: bool


def create_app(search_service: SearchService | None = None) -> FastAPI:
    """Create the API; optional model weights load only on semantic requests."""
    application = FastAPI(
        title="CodeSearch-FT API",
        description="Local API foundation for Python code search.",
        version=__version__,
    )
    application.state.search_service = search_service

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        del request, exc
        return _error(422, "validation_error", "Search request validation failed.")

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Report process liveness separately from future search readiness."""
        return HealthResponse(
            status="ok",
            service="codesearch-ft",
            version=__version__,
            search_ready=search_service is not None,
        )

    @application.get("/v1/models", response_model=list[ModelResponse], tags=["models"])
    def models() -> list[ModelResponse]:
        """List supported model identifiers without inspecting caches or downloading."""
        return [
            ModelResponse(
                model_id=model.model_id,
                role=model.role,
                source=model.source,
                weights_checked=model.weights_checked,
            )
            for model in list_configured_models()
        ]

    @application.get("/v1/indexes", response_model=list[IndexResponse], tags=["indexes"])
    def indexes() -> list[IndexResponse]:
        """List semantic index metadata without exposing local artifact paths."""
        if search_service is None:
            return []
        return [
            IndexResponse(
                index_id=item.index_id,
                repository_id=item.repository_id,
                model_id=item.model_id,
                model_revision=item.model_revision,
                preprocessing_version=item.preprocessing_version,
                embedding_dimension=item.embedding_dimension,
                vector_count=item.vector_count,
                status=item.status,
                created_at=item.created_at,
            )
            for item in search_service.list_indexes()
        ]

    @application.post(
        "/v1/search",
        response_model=SearchResponse,
        responses={
            422: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["search"],
    )
    def search(request: SearchRequest) -> SearchResponse | JSONResponse:
        """Execute one bounded local query and return source-oriented results."""
        if search_service is None:
            return _error(503, "search_unavailable", "Search service is not configured.")
        started = perf_counter()
        try:
            outcome = search_service.search(
                request.query,
                request.mode,
                request.top_k,
                request.index_id,
                request.filters.repository_id if request.filters else None,
            )
        except IndexNotFoundError:
            return _error(404, "index_not_found", "The requested semantic index does not exist.")
        except ModelUnavailableError:
            return _error(
                503,
                "model_unavailable",
                "Model dependencies or locally cached weights are unavailable. "
                "Use the explicit model download command or select lexical mode.",
            )
        except IndexCompatibilityError:
            return _error(
                409,
                "index_incompatible",
                "The selected semantic index is unavailable or incompatible; rebuild it and retry.",
            )
        except CodeSearchError:
            logger.exception("Search failed in a local adapter")
            return _error(503, "search_failed", "The local search operation could not complete.")
        except ValueError as exc:
            return _error(422, "invalid_search", str(exc))
        elapsed_ms = (perf_counter() - started) * 1_000
        return SearchResponse(
            query=request.query,
            model_id=outcome.model_id,
            index_id=outcome.index_id,
            retrieval_mode=request.mode,
            elapsed_ms=elapsed_ms,
            results=[
                SearchResultResponse(
                    function_id=result.function_id,
                    name=result.name,
                    qualified_name=result.qualified_name,
                    file_path=result.file_path,
                    line_start=result.line_start,
                    line_end=result.line_end,
                    code_snippet=result.code_snippet,
                    rank=result.rank,
                    score=result.score,
                    score_type=result.score_type,
                )
                for result in outcome.results
            ],
        )

    return application


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )
