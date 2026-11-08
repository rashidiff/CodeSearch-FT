"""Lexical BM25 retrieval application adapter."""

from collections.abc import Sequence

from codesearch_ft.domain.errors import StorageError
from codesearch_ft.domain.models import SearchHit
from codesearch_ft.domain.ports import MetadataRepository


class FTS5LexicalRetriever:
    """Validate bounds and delegate safe tokenized lexical search to SQLite."""

    def __init__(self, repository: MetadataRepository) -> None:
        self._repository = repository

    def search(
        self, query: str, top_k: int = 10, repository_id: int | None = None
    ) -> Sequence[SearchHit]:
        if not query.strip():
            return ()
        if len(query) > 2_000:
            raise StorageError("Search query exceeds the 2,000 character limit.")
        if not 1 <= top_k <= 1_000:
            raise StorageError("top_k must be between 1 and 1000.")
        return self._repository.search_lexical(query, top_k, repository_id)
