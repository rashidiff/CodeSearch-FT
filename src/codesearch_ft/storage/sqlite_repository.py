"""SQLite metadata adapter for local repositories and lexical search."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Concatenate, ParamSpec, TypeVar

from codesearch_ft.domain.errors import IndexCompatibilityError, StorageError
from codesearch_ft.domain.identity import corpus_fingerprint
from codesearch_ft.domain.models import (
    ExtractedFunction,
    IndexedFunction,
    IndexVersion,
    RetrievalMode,
    SearchHit,
    VectorMatch,
)
from codesearch_ft.storage.migrations import migrate

P = ParamSpec("P")
T = TypeVar("T")


def _serialized(
    method: Callable[Concatenate[SQLiteRepository, P], T],
) -> Callable[Concatenate[SQLiteRepository, P], T]:
    @wraps(method)
    def wrapped(self: SQLiteRepository, *args: P.args, **kwargs: P.kwargs) -> T:
        with self.serialized_lock:
            return method(self, *args, **kwargs)

    return wrapped


class SQLiteRepository:
    """Own one SQLite connection and keep schema/FTS changes transactional."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.serialized_lock = RLock()
        if str(database_path) != ":memory:":
            try:
                database_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StorageError(
                    f"Unable to create database directory {database_path.parent}: {exc}"
                ) from exc
        try:
            self._connection = sqlite3.connect(database_path, timeout=30, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            migrate(self._connection)
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to open SQLite database at {database_path}: {exc}") from exc

    @_serialized
    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SQLiteRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @_serialized
    def get_or_create_repository(self, root_path: Path) -> int:
        root = str(root_path.resolve())
        now = _now()
        with self._connection:
            self._connection.execute(
                "INSERT INTO repositories(root_path, created_at, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(root_path) DO UPDATE SET updated_at=excluded.updated_at",
                (root, now, now),
            )
            row = self._connection.execute(
                "SELECT id FROM repositories WHERE root_path = ?", (root,)
            ).fetchone()
        if row is None:
            raise StorageError("Unable to create or load repository metadata.")
        return int(row["id"])

    @_serialized
    def file_hash(self, repository_id: int, relative_path: str) -> str | None:
        row = self._connection.execute(
            "SELECT sha256 FROM source_files WHERE repository_id = ? AND relative_path = ?",
            (repository_id, relative_path),
        ).fetchone()
        return str(row["sha256"]) if row else None

    @_serialized
    def replace_file_functions(
        self,
        repository_id: int,
        relative_path: str,
        file_sha256: str,
        functions: Sequence[ExtractedFunction],
    ) -> None:
        now = _now()
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO source_files("
                    "repository_id, relative_path, sha256, indexed_at, status"
                    ") VALUES (?, ?, ?, ?, 'indexed') "
                    "ON CONFLICT(repository_id, relative_path) DO UPDATE SET "
                    "sha256=excluded.sha256, indexed_at=excluded.indexed_at, "
                    "status='indexed'",
                    (repository_id, relative_path, file_sha256, now),
                )
                file_row = self._connection.execute(
                    "SELECT id FROM source_files WHERE repository_id=? AND relative_path=?",
                    (repository_id, relative_path),
                ).fetchone()
                if file_row is None:
                    raise StorageError(f"Missing source file row for {relative_path}.")
                file_id = int(file_row["id"])
                active_keys = {function.symbol_key for function in functions}
                for function in functions:
                    self._connection.execute(
                        "INSERT INTO functions("
                        "source_file_id, symbol_key, name, qualified_name, signature, "
                        "source_code, embedding_input, docstring, start_line, end_line, "
                        "content_sha256) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(source_file_id, symbol_key) DO UPDATE SET "
                        "name=excluded.name, qualified_name=excluded.qualified_name, "
                        "signature=excluded.signature, source_code=excluded.source_code, "
                        "embedding_input=excluded.embedding_input, docstring=excluded.docstring, "
                        "start_line=excluded.start_line, end_line=excluded.end_line, "
                        "content_sha256=excluded.content_sha256",
                        (
                            file_id,
                            function.symbol_key,
                            function.name,
                            function.qualified_name,
                            function.signature,
                            function.source_code,
                            function.embedding_input,
                            function.docstring,
                            function.line_start,
                            function.line_end,
                            function.content_sha256,
                        ),
                    )
                rows = self._connection.execute(
                    "SELECT id, symbol_key FROM functions WHERE source_file_id = ?", (file_id,)
                ).fetchall()
                stale_ids = [int(row["id"]) for row in rows if row["symbol_key"] not in active_keys]
                if stale_ids:
                    self._connection.executemany(
                        "DELETE FROM functions WHERE id = ?",
                        ((function_id,) for function_id in stale_ids),
                    )
                self._mark_ready_indexes_stale(repository_id)
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to update indexed file {relative_path}: {exc}") from exc

    @_serialized
    def remove_source_file(self, repository_id: int, relative_path: str) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    "DELETE FROM source_files WHERE repository_id=? AND relative_path=?",
                    (repository_id, relative_path),
                )
                self._mark_ready_indexes_stale(repository_id)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Unable to remove stale source file {relative_path}: {exc}"
            ) from exc

    @_serialized
    def record_file_error(self, repository_id: int, relative_path: str, file_sha256: str) -> None:
        """Record a parse failure and atomically remove results from old source."""
        now = _now()
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO source_files("
                    "repository_id, relative_path, sha256, indexed_at, status"
                    ") VALUES (?, ?, ?, ?, 'error') "
                    "ON CONFLICT(repository_id, relative_path) DO UPDATE SET "
                    "sha256=excluded.sha256, indexed_at=excluded.indexed_at, status='error'",
                    (repository_id, relative_path, file_sha256, now),
                )
                row = self._connection.execute(
                    "SELECT id FROM source_files WHERE repository_id=? AND relative_path=?",
                    (repository_id, relative_path),
                ).fetchone()
                if row is None:
                    raise StorageError(f"Missing source file row for {relative_path}.")
                self._connection.execute(
                    "DELETE FROM functions WHERE source_file_id = ?", (int(row["id"]),)
                )
                self._mark_ready_indexes_stale(repository_id)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Unable to record parse failure for {relative_path}: {exc}"
            ) from exc

    @_serialized
    def remove_unseen_files(self, repository_id: int, seen_paths: set[str]) -> int:
        rows = self._connection.execute(
            "SELECT relative_path FROM source_files WHERE repository_id=?", (repository_id,)
        ).fetchall()
        stale = [
            str(row["relative_path"]) for row in rows if str(row["relative_path"]) not in seen_paths
        ]
        try:
            with self._connection:
                self._connection.executemany(
                    "DELETE FROM source_files WHERE repository_id=? AND relative_path=?",
                    ((repository_id, path) for path in stale),
                )
                if stale:
                    self._mark_ready_indexes_stale(repository_id)
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to remove stale repository files: {exc}") from exc
        return len(stale)

    @_serialized
    def search_lexical(
        self, query: str, top_k: int, repository_id: int | None = None
    ) -> tuple[SearchHit, ...]:
        terms = _query_terms(query)
        if not terms:
            return ()
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        try:
            rows = self._connection.execute(
                "SELECT functions.id, functions.name, functions.qualified_name, "
                "repositories.root_path, source_files.relative_path, functions.start_line, "
                "functions.end_line, functions.source_code, "
                "bm25(function_fts, 3.0, 2.0, 1.5, 1.0, 1.0) AS score "
                "FROM function_fts JOIN functions ON functions.id=function_fts.rowid "
                "JOIN source_files ON source_files.id=functions.source_file_id "
                "JOIN repositories ON repositories.id=source_files.repository_id "
                "WHERE function_fts MATCH ? "
                "AND (? IS NULL OR repositories.id = ?) "
                "ORDER BY score ASC, functions.id ASC LIMIT ?",
                (match_query, repository_id, repository_id, top_k),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Lexical query failed safely: {exc}") from exc
        return tuple(
            SearchHit(
                function_id=str(row["id"]),
                name=str(row["name"]),
                qualified_name=str(row["qualified_name"]),
                file_path=str(Path(str(row["root_path"])) / str(row["relative_path"])),
                line_start=int(row["start_line"]),
                line_end=int(row["end_line"]),
                code_snippet=str(row["source_code"]),
                retrieval_mode=RetrievalMode.LEXICAL,
                rank=rank,
                score=float(row["score"]),
                score_type="sqlite_fts5_bm25 (lower is better)",
            )
            for rank, row in enumerate(rows, start=1)
        )

    @_serialized
    def list_indexed_functions(self, repository_id: int) -> tuple[IndexedFunction, ...]:
        rows = self._connection.execute(
            "SELECT functions.*, source_files.relative_path FROM functions "
            "JOIN source_files ON source_files.id=functions.source_file_id "
            "WHERE source_files.repository_id=? ORDER BY functions.id",
            (repository_id,),
        ).fetchall()
        return tuple(
            IndexedFunction(
                function_id=int(row["id"]),
                function=ExtractedFunction(
                    symbol_key=str(row["symbol_key"]),
                    name=str(row["name"]),
                    qualified_name=str(row["qualified_name"]),
                    signature=str(row["signature"]),
                    file_path=str(row["relative_path"]),
                    line_start=int(row["start_line"]),
                    line_end=int(row["end_line"]),
                    source_code=str(row["source_code"]),
                    embedding_input=str(row["embedding_input"]),
                    content_sha256=str(row["content_sha256"]),
                    docstring=str(row["docstring"]) if row["docstring"] is not None else None,
                ),
            )
            for row in rows
        )

    @_serialized
    def create_index_version(
        self,
        repository_id: int,
        model_id: str,
        model_revision: str,
        preprocessing_version: str,
        embedding_dimension: int,
        corpus_sha256: str,
        index_path: Path,
    ) -> IndexVersion:
        try:
            with self._connection:
                cursor = self._connection.execute(
                    "INSERT INTO index_versions("
                    "repository_id, model_id, model_revision, preprocessing_version, "
                    "embedding_dimension, corpus_sha256, index_path, status, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, 'building', ?)",
                    (
                        repository_id,
                        model_id,
                        model_revision,
                        preprocessing_version,
                        embedding_dimension,
                        corpus_sha256,
                        str(index_path),
                        _now(),
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM index_versions WHERE id=?", (cursor.lastrowid,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to register semantic index build: {exc}") from exc
        if row is None:
            raise StorageError("SQLite did not return the new semantic index version.")
        return _index_version(row)

    @_serialized
    def fail_index_version(self, index_id: int) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    "DELETE FROM vector_mapping WHERE index_version_id=?", (index_id,)
                )
                self._connection.execute(
                    "UPDATE index_versions SET status='failed' WHERE id=? AND status='building'",
                    (index_id,),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to mark semantic index {index_id} failed: {exc}") from exc

    @_serialized
    def add_vector_mappings(self, index_id: int, mappings: Sequence[tuple[int, int]]) -> None:
        if not mappings:
            return
        try:
            with self._connection:
                version = self._connection.execute(
                    "SELECT repository_id FROM index_versions WHERE id=? AND status='building'",
                    (index_id,),
                ).fetchone()
                if version is None:
                    raise IndexCompatibilityError(
                        f"Semantic index {index_id} is not in the building state."
                    )
                repository_id = int(version["repository_id"])
                function_ids = tuple(function_id for function_id, _ in mappings)
                placeholders = ",".join("?" for _ in function_ids)
                rows = self._connection.execute(
                    "SELECT functions.id FROM functions "
                    "JOIN source_files ON source_files.id=functions.source_file_id "
                    f"WHERE source_files.repository_id=? AND functions.id IN ({placeholders})",
                    (repository_id, *function_ids),
                ).fetchall()
                if {int(row["id"]) for row in rows} != set(function_ids):
                    raise IndexCompatibilityError(
                        "Vector mappings contain functions outside the target repository."
                    )
                self._connection.executemany(
                    "INSERT INTO vector_mapping(function_id, index_version_id, vector_id) "
                    "VALUES (?, ?, ?)",
                    ((function_id, index_id, vector_id) for function_id, vector_id in mappings),
                )
        except sqlite3.Error as exc:
            raise StorageError(
                f"Unable to record vector mappings for index {index_id}: {exc}"
            ) from exc

    @_serialized
    def promote_index_version(
        self, index_id: int, artifact_sha256: str, vector_count: int
    ) -> IndexVersion:
        if vector_count < 0:
            raise ValueError("vector_count must be non-negative")
        try:
            with self._connection:
                row = self._connection.execute(
                    "SELECT repository_id FROM index_versions WHERE id=? AND status='building'",
                    (index_id,),
                ).fetchone()
                if row is None:
                    raise IndexCompatibilityError(
                        f"Semantic index version {index_id} is not in the building state."
                    )
                repository_id = int(row["repository_id"])
                current_corpus = self._corpus_sha256(repository_id)
                pending = self._connection.execute(
                    "SELECT corpus_sha256 FROM index_versions WHERE id=?", (index_id,)
                ).fetchone()
                if pending is None or str(pending["corpus_sha256"]) != current_corpus:
                    raise IndexCompatibilityError(
                        "Repository functions changed while the semantic index was being built; "
                        "the staged index was not promoted. Rebuild and retry."
                    )
                mapping_count = int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM vector_mapping WHERE index_version_id=?",
                        (index_id,),
                    ).fetchone()[0]
                )
                if mapping_count != vector_count:
                    raise IndexCompatibilityError(
                        f"Semantic index {index_id} has {vector_count} vectors but "
                        f"{mapping_count} SQLite mappings."
                    )
                self._connection.execute(
                    "UPDATE index_versions SET status='stale' "
                    "WHERE repository_id=? AND status='ready'",
                    (repository_id,),
                )
                self._connection.execute(
                    "UPDATE index_versions SET status='ready', artifact_sha256=?, vector_count=? "
                    "WHERE id=? AND status='building'",
                    (artifact_sha256, vector_count, index_id),
                )
                result = self._connection.execute(
                    "SELECT * FROM index_versions WHERE id=?", (index_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to promote semantic index {index_id}: {exc}") from exc
        if result is None:
            raise StorageError(f"Promoted semantic index {index_id} is unavailable.")
        return _index_version(result)

    @_serialized
    def get_index_version(self, index_id: int) -> IndexVersion | None:
        row = self._connection.execute(
            "SELECT * FROM index_versions WHERE id=?", (index_id,)
        ).fetchone()
        return _index_version(row) if row else None

    @_serialized
    def list_index_versions(self, status: str | None = None) -> tuple[IndexVersion, ...]:
        if status is None:
            rows = self._connection.execute(
                "SELECT * FROM index_versions ORDER BY id DESC"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM index_versions WHERE status=? ORDER BY id DESC", (status,)
            ).fetchall()
        return tuple(_index_version(row) for row in rows)

    @_serialized
    def resolve_vector_matches(
        self, index_id: int, matches: Sequence[VectorMatch]
    ) -> tuple[SearchHit, ...]:
        hits: list[SearchHit] = []
        for rank, match in enumerate(matches, start=1):
            row = self._connection.execute(
                "SELECT functions.id, functions.name, functions.qualified_name, "
                "repositories.root_path, source_files.relative_path, functions.start_line, "
                "functions.end_line, functions.source_code "
                "FROM vector_mapping "
                "JOIN index_versions ON index_versions.id=vector_mapping.index_version_id "
                "JOIN functions ON functions.id=vector_mapping.function_id "
                "JOIN source_files ON source_files.id=functions.source_file_id "
                "JOIN repositories ON repositories.id=source_files.repository_id "
                "WHERE vector_mapping.index_version_id=? AND vector_mapping.vector_id=? "
                "AND index_versions.status='ready'",
                (index_id, match.vector_id),
            ).fetchone()
            if row is None:
                raise IndexCompatibilityError(
                    f"Vector {match.vector_id} in semantic index {index_id} "
                    "has no active function mapping."
                )
            hits.append(
                SearchHit(
                    function_id=str(row["id"]),
                    name=str(row["name"]),
                    qualified_name=str(row["qualified_name"]),
                    file_path=str(Path(str(row["root_path"])) / str(row["relative_path"])),
                    line_start=int(row["start_line"]),
                    line_end=int(row["end_line"]),
                    code_snippet=str(row["source_code"]),
                    retrieval_mode=RetrievalMode.SEMANTIC,
                    rank=rank,
                    score=match.score,
                    score_type="cosine_similarity",
                )
            )
        return tuple(hits)

    @_serialized
    def validate_vector_ids(self, index_id: int, vector_ids: Sequence[int]) -> None:
        rows = self._connection.execute(
            "SELECT vector_id FROM vector_mapping WHERE index_version_id=? ORDER BY vector_id",
            (index_id,),
        ).fetchall()
        mapped = tuple(int(row["vector_id"]) for row in rows)
        if tuple(sorted(vector_ids)) != mapped:
            raise IndexCompatibilityError(
                f"FAISS vector IDs do not match SQLite mappings for semantic index {index_id}."
            )

    def _mark_ready_indexes_stale(self, repository_id: int) -> None:
        self._connection.execute(
            "UPDATE index_versions SET status='stale' WHERE repository_id=? AND status='ready'",
            (repository_id,),
        )

    def _corpus_sha256(self, repository_id: int) -> str:
        rows = self._connection.execute(
            "SELECT functions.id, functions.content_sha256 FROM functions "
            "JOIN source_files ON source_files.id=functions.source_file_id "
            "WHERE source_files.repository_id=? ORDER BY functions.id",
            (repository_id,),
        ).fetchall()
        return corpus_fingerprint((int(row["id"]), str(row["content_sha256"])) for row in rows)


def _query_terms(query: str) -> tuple[str, ...]:
    """Treat user text as words, not FTS5 operators or raw query syntax."""
    terms: list[str] = []
    current: list[str] = []
    for character in query:
        if character.isalnum() or character == "_":
            current.append(character)
        elif current:
            terms.append("".join(current))
            current.clear()
    if current:
        terms.append("".join(current))
    return tuple(terms)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _index_version(row: sqlite3.Row) -> IndexVersion:
    return IndexVersion(
        index_id=int(row["id"]),
        repository_id=int(row["repository_id"]),
        model_id=str(row["model_id"]),
        model_revision=str(row["model_revision"]),
        preprocessing_version=str(row["preprocessing_version"]),
        embedding_dimension=int(row["embedding_dimension"]),
        index_path=str(row["index_path"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        artifact_sha256=str(row["artifact_sha256"]),
        vector_count=int(row["vector_count"]),
        corpus_sha256=str(row["corpus_sha256"]),
    )
