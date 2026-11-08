"""Staged, versioned FAISS index builds and artifact compatibility validation."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from codesearch_ft.domain.errors import IndexCompatibilityError
from codesearch_ft.domain.identity import corpus_fingerprint
from codesearch_ft.domain.models import IndexVersion
from codesearch_ft.domain.ports import CodeEncoder, MetadataRepository
from codesearch_ft.retrieval.faiss_index import FaissFlatIPIndex


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Integrity and provenance record stored beside a persisted FAISS file."""

    manifest_version: int
    index_id: int
    repository_id: int
    model_id: str
    model_revision: str
    preprocessing_version: str
    embedding_dimension: int
    vector_count: int
    corpus_sha256: str
    artifact_sha256: str


class SemanticIndexManager:
    """Build complete replacement indexes and promote only validated artifacts."""

    _process_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(
        self,
        repository: MetadataRepository,
        artifact_root: Path,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Indexing batch size must be positive.")
        self._repository = repository
        self._artifact_root = artifact_root.resolve()
        self._batch_size = batch_size

    def rebuild(
        self,
        repository_id: int,
        encoder: CodeEncoder,
        *,
        preprocessing_version: str,
    ) -> IndexVersion:
        """Embed the current repository snapshot and atomically promote its index."""
        if encoder.dimension < 1:
            raise ValueError("Encoder dimension must be positive.")
        if not encoder.model_id or not encoder.model_revision or not preprocessing_version:
            raise ValueError(
                "Model ID, immutable revision, and preprocessing version are required."
            )

        with self._process_lock:
            functions = tuple(self._repository.list_indexed_functions(repository_id))
            corpus_sha256 = corpus_fingerprint(
                (function.function_id, function.function.content_sha256) for function in functions
            )
            repository_dir = self._artifact_root / f"repository-{repository_id}"
            repository_dir.mkdir(parents=True, exist_ok=True)
            artifact_id = uuid.uuid4().hex
            index_path = repository_dir / f"semantic-{artifact_id}.faiss"
            manifest_path = _manifest_path(index_path)
            stage_path = repository_dir / f".semantic-{artifact_id}.staging.faiss"
            stage_manifest = repository_dir / f".semantic-{artifact_id}.staging.manifest.json"
            version = self._repository.create_index_version(
                repository_id=repository_id,
                model_id=encoder.model_id,
                model_revision=encoder.model_revision,
                preprocessing_version=preprocessing_version,
                embedding_dimension=encoder.dimension,
                corpus_sha256=corpus_sha256,
                index_path=index_path,
            )
            try:
                index = FaissFlatIPIndex(encoder.dimension)
                for start in range(0, len(functions), self._batch_size):
                    batch = functions[start : start + self._batch_size]
                    embeddings = encoder.encode_code(
                        tuple(item.function.embedding_input for item in batch)
                    )
                    if len(embeddings) != len(batch):
                        raise IndexCompatibilityError(
                            "Encoder returned a different number of vectors than input functions."
                        )
                    index.add(
                        tuple(item.function_id for item in batch),
                        embeddings,
                    )
                    self._repository.add_vector_mappings(
                        version.index_id,
                        tuple((item.function_id, item.function_id) for item in batch),
                    )
                index.save(stage_path)
                artifact_sha256 = _sha256_file(stage_path)
                manifest = IndexManifest(
                    manifest_version=1,
                    index_id=version.index_id,
                    repository_id=repository_id,
                    model_id=encoder.model_id,
                    model_revision=encoder.model_revision,
                    preprocessing_version=preprocessing_version,
                    embedding_dimension=encoder.dimension,
                    vector_count=index.count,
                    corpus_sha256=corpus_sha256,
                    artifact_sha256=artifact_sha256,
                )
                _write_manifest(stage_manifest, manifest)
                _validate_index_metadata(index, manifest)
                os.replace(stage_path, index_path)
                os.replace(stage_manifest, manifest_path)
                return self._repository.promote_index_version(
                    version.index_id, artifact_sha256, index.count
                )
            except Exception:
                self._repository.fail_index_version(version.index_id)
                stage_path.unlink(missing_ok=True)
                stage_manifest.unlink(missing_ok=True)
                index_path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
                raise

    def load(self, version: IndexVersion) -> FaissFlatIPIndex:
        """Load a ready index only after validating artifact and database mappings."""
        current_version = self._repository.get_index_version(version.index_id)
        if current_version is None:
            raise IndexCompatibilityError(
                f"Semantic index {version.index_id} is not registered in SQLite."
            )
        version = current_version
        if version.status != "ready":
            raise IndexCompatibilityError(
                f"Semantic index {version.index_id} is {version.status}; "
                "rebuild it before searching."
            )
        path = Path(version.index_path).resolve()
        try:
            path.relative_to(self._artifact_root)
        except ValueError as exc:
            raise IndexCompatibilityError(
                "Semantic index path escapes the configured artifact root."
            ) from exc
        manifest = _read_manifest(_manifest_path(path))
        _validate_version(version, manifest)
        artifact_sha256 = _sha256_file(path)
        if (
            artifact_sha256 != manifest.artifact_sha256
            or artifact_sha256 != version.artifact_sha256
        ):
            raise IndexCompatibilityError(
                f"Semantic index {version.index_id} failed its SHA-256 integrity check."
            )
        index = FaissFlatIPIndex(version.embedding_dimension)
        index.load(path)
        _validate_index_metadata(index, manifest)
        if index.count != version.vector_count:
            raise IndexCompatibilityError(
                f"Semantic index {version.index_id} has {index.count} vectors; "
                f"SQLite records {version.vector_count}."
            )
        self._repository.validate_vector_ids(version.index_id, index.vector_ids)
        return index


def _manifest_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.name}.manifest.json")


def _write_manifest(path: Path, manifest: IndexManifest) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(asdict(manifest), output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _read_manifest(path: Path) -> IndexManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("manifest must be a JSON object")
        return IndexManifest(**cast(dict[str, Any], raw))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IndexCompatibilityError(
            f"Semantic index manifest is missing or malformed: {exc}"
        ) from exc


def _validate_version(version: IndexVersion, manifest: IndexManifest) -> None:
    expected = (
        version.index_id,
        version.repository_id,
        version.model_id,
        version.model_revision,
        version.preprocessing_version,
        version.embedding_dimension,
        version.vector_count,
        version.corpus_sha256,
        version.artifact_sha256,
    )
    actual = (
        manifest.index_id,
        manifest.repository_id,
        manifest.model_id,
        manifest.model_revision,
        manifest.preprocessing_version,
        manifest.embedding_dimension,
        manifest.vector_count,
        manifest.corpus_sha256,
        manifest.artifact_sha256,
    )
    if manifest.manifest_version != 1 or actual != expected:
        raise IndexCompatibilityError(
            f"Semantic index {version.index_id} manifest does not match SQLite metadata."
        )


def _validate_index_metadata(index: FaissFlatIPIndex, manifest: IndexManifest) -> None:
    if index.dimension != manifest.embedding_dimension or index.count != manifest.vector_count:
        raise IndexCompatibilityError(
            "FAISS artifact dimension or vector count does not match its manifest."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IndexCompatibilityError(
            f"Unable to read semantic index artifact {path}: {exc}"
        ) from exc
    return digest.hexdigest()
