"""Exact FAISS CPU index using inner product over normalized vectors."""

from __future__ import annotations

import importlib
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from codesearch_ft.domain.errors import IndexCompatibilityError, ModelUnavailableError
from codesearch_ft.domain.models import Embedding, VectorMatch


class FaissFlatIPIndex:
    """A CPU-only `IndexIDMap2(IndexFlatIP)` with normalized float32 vectors."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("FAISS embedding dimension must be positive.")
        self.dimension = dimension
        self._faiss, self._numpy = _faiss_runtime()
        self._index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dimension))
        self._vector_ids: set[int] = set()

    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    @property
    def vector_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._vector_ids))

    def add(self, vector_ids: Sequence[int], vectors: Sequence[Embedding]) -> None:
        if len(vector_ids) != len(vectors):
            raise ValueError("Vector IDs and embedding rows must have equal lengths.")
        if not vectors:
            return
        if len(set(vector_ids)) != len(vector_ids) or any(
            vector_id < 0 for vector_id in vector_ids
        ):
            raise ValueError("FAISS vector IDs must be unique non-negative integers.")
        if self._vector_ids.intersection(vector_ids):
            raise ValueError("FAISS vector IDs cannot be added more than once.")
        matrix = self._validated_matrix(vectors)
        norms = self._numpy.linalg.norm(matrix, axis=1, keepdims=True)
        if self._numpy.any(norms <= 1e-12):
            raise ValueError("FAISS cannot index a zero-norm embedding.")
        matrix = matrix / norms
        ids = self._numpy.asarray(vector_ids, dtype=self._numpy.int64)
        self._index.add_with_ids(matrix, ids)
        self._vector_ids.update(int(vector_id) for vector_id in vector_ids)

    def search(self, query: Embedding, top_k: int) -> tuple[VectorMatch, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        matrix = self._validated_matrix((query,))
        norm = float(self._numpy.linalg.norm(matrix[0]))
        if norm <= 1e-12:
            raise ValueError("FAISS query embedding must have non-zero norm.")
        scores, ids = self._index.search(matrix / norm, top_k)
        return tuple(
            VectorMatch(vector_id=int(vector_id), score=float(score))
            for score, vector_id in zip(scores[0], ids[0], strict=True)
            if int(vector_id) >= 0
        )

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
        try:
            self._faiss.write_index(self._index, str(staging))
            os.replace(staging, destination)
        except Exception as exc:
            staging.unlink(missing_ok=True)
            raise IndexCompatibilityError(
                f"Unable to persist FAISS index at {destination}: {exc}"
            ) from exc

    def load(self, source: Path) -> None:
        try:
            loaded = self._faiss.read_index(str(source))
            if not isinstance(loaded, self._faiss.IndexIDMap2):
                raise IndexCompatibilityError(
                    "Persisted FAISS artifact is not an IndexIDMap2 index."
                )
            if not isinstance(self._faiss.downcast_index(loaded.index), self._faiss.IndexFlatIP):
                raise IndexCompatibilityError(
                    "Persisted FAISS artifact is not an exact IndexFlatIP index."
                )
            if loaded.d != self.dimension or loaded.metric_type != self._faiss.METRIC_INNER_PRODUCT:
                raise IndexCompatibilityError(
                    f"Persisted FAISS index is incompatible: dimension={loaded.d}, "
                    f"metric={loaded.metric_type}; expected dimension={self.dimension} "
                    "inner product."
                )
            ids = self._faiss.vector_to_array(loaded.id_map).astype(self._numpy.int64, copy=False)
            if len(set(int(vector_id) for vector_id in ids.tolist())) != len(ids):
                raise IndexCompatibilityError(
                    "Persisted FAISS index contains duplicate vector IDs."
                )
        except IndexCompatibilityError:
            raise
        except Exception as exc:
            raise IndexCompatibilityError(f"Unable to load FAISS index {source}: {exc}") from exc
        self._index = loaded
        self._vector_ids = {int(vector_id) for vector_id in ids.tolist()}

    def _validated_matrix(self, vectors: Sequence[Embedding]) -> Any:
        matrix = self._numpy.asarray(vectors, dtype=self._numpy.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding matrix must have shape [batch, {self.dimension}], got {matrix.shape}."
            )
        if not self._numpy.all(self._numpy.isfinite(matrix)):
            raise ValueError("Embeddings must contain only finite values.")
        return matrix


def _faiss_runtime() -> tuple[Any, Any]:
    try:
        return importlib.import_module("faiss"), importlib.import_module("numpy")
    except ImportError as exc:
        raise ModelUnavailableError(
            "FAISS CPU dependencies are missing. Install CodeSearch-FT inference dependencies "
            "with `uv sync --extra inference`."
        ) from exc
