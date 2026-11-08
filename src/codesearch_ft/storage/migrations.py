"""Numbered, transactional SQLite schema migrations."""

import sqlite3
from collections.abc import Callable

from codesearch_ft.domain.errors import StorageError

Migration = tuple[int, Callable[[sqlite3.Connection], None]]


def _migration_1(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        BEGIN IMMEDIATE;
        CREATE TABLE repositories (
            id INTEGER PRIMARY KEY,
            root_path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE source_files (
            id INTEGER PRIMARY KEY,
            repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
            relative_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('indexed', 'skipped', 'error')),
            UNIQUE (repository_id, relative_path)
        );
        CREATE INDEX idx_source_files_repository ON source_files(repository_id);
        CREATE TABLE functions (
            id INTEGER PRIMARY KEY,
            source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
            symbol_key TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            signature TEXT NOT NULL,
            source_code TEXT NOT NULL,
            embedding_input TEXT NOT NULL,
            docstring TEXT,
            start_line INTEGER NOT NULL CHECK (start_line > 0),
            end_line INTEGER NOT NULL CHECK (end_line >= start_line),
            content_sha256 TEXT NOT NULL,
            UNIQUE (source_file_id, symbol_key)
        );
        CREATE INDEX idx_functions_qualified_name ON functions(qualified_name);
        CREATE TABLE index_versions (
            id INTEGER PRIMARY KEY,
            repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            preprocessing_version TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
            index_path TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('building', 'ready', 'failed', 'stale')),
            created_at TEXT NOT NULL
        );
        CREATE TABLE vector_mapping (
            function_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
            index_version_id INTEGER NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
            vector_id INTEGER NOT NULL,
            PRIMARY KEY (function_id, index_version_id),
            UNIQUE (index_version_id, vector_id)
        );
        CREATE VIRTUAL TABLE function_fts USING fts5(
            name, qualified_name, signature, docstring, source_code,
            content='functions', content_rowid='id', tokenize='unicode61'
        );
        CREATE TRIGGER functions_ai AFTER INSERT ON functions BEGIN
            INSERT INTO function_fts(
                rowid, name, qualified_name, signature, docstring, source_code
            ) VALUES (
                new.id, new.name, new.qualified_name, new.signature, new.docstring, new.source_code
            );
        END;
        CREATE TRIGGER functions_ad AFTER DELETE ON functions BEGIN
            INSERT INTO function_fts(
                function_fts, rowid, name, qualified_name, signature, docstring, source_code
            ) VALUES (
                'delete', old.id, old.name, old.qualified_name,
                old.signature, old.docstring, old.source_code
            );
        END;
        CREATE TRIGGER functions_au AFTER UPDATE ON functions BEGIN
            INSERT INTO function_fts(
                function_fts, rowid, name, qualified_name, signature, docstring, source_code
            ) VALUES (
                'delete', old.id, old.name, old.qualified_name,
                old.signature, old.docstring, old.source_code
            );
            INSERT INTO function_fts(
                rowid, name, qualified_name, signature, docstring, source_code
            ) VALUES (
                new.id, new.name, new.qualified_name, new.signature, new.docstring, new.source_code
            );
        END;
        """
    )


def _migration_2(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE index_versions ADD COLUMN artifact_sha256 TEXT NOT NULL DEFAULT ''"
    )
    connection.execute(
        "ALTER TABLE index_versions ADD COLUMN vector_count INTEGER NOT NULL DEFAULT 0 "
        "CHECK (vector_count >= 0)"
    )
    connection.execute(
        "ALTER TABLE index_versions ADD COLUMN corpus_sha256 TEXT NOT NULL DEFAULT ''"
    )
    connection.execute(
        "CREATE UNIQUE INDEX idx_one_ready_index_per_repository "
        "ON index_versions(repository_id) WHERE status = 'ready'"
    )


MIGRATIONS: tuple[Migration, ...] = ((1, _migration_1), (2, _migration_2))


def migrate(connection: sqlite3.Connection) -> None:
    """Apply each pending migration atomically and record its version."""
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        applied = {
            int(row[0]) for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for version, apply in MIGRATIONS:
            if version in applied:
                continue
            connection.execute("BEGIN IMMEDIATE")
            try:
                apply(connection)
                connection.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
    except (sqlite3.Error, RuntimeError) as exc:
        raise StorageError(f"SQLite schema migration failed: {exc}") from exc
