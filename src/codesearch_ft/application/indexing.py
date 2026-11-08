"""Orchestration for initial and incremental local repository indexing."""

from dataclasses import dataclass
from pathlib import Path

from codesearch_ft.domain.errors import IndexingError
from codesearch_ft.domain.ports import CodeParser, MetadataRepository
from codesearch_ft.ingestion.repository_scanner import RepositoryScanner, ScanIssue


@dataclass(frozen=True, slots=True)
class IndexSummary:
    """Counts from one complete or partial repository indexing run."""

    repository_id: int
    discovered_files: int
    indexed_files: int
    unchanged_files: int
    removed_files: int
    function_count: int
    issues: tuple[ScanIssue, ...]
    complete: bool


class RepositoryIndexingService:
    """Scan source, parse changed files, and reconcile metadata without execution."""

    def __init__(
        self,
        scanner: RepositoryScanner,
        parser: CodeParser,
        repository: MetadataRepository,
    ) -> None:
        self._scanner = scanner
        self._parser = parser
        self._repository = repository

    def index(self, root: Path, update: bool = False) -> IndexSummary:
        scan = self._scanner.scan(root)
        resolved_root = root.expanduser().resolve(strict=True)
        repository_id = self._repository.get_or_create_repository(resolved_root)
        if not scan.complete:
            raise IndexingError(
                "Repository scan is incomplete; metadata was preserved. "
                "Review access and configured limits, then retry."
            )
        indexed = unchanged = function_count = 0
        seen_paths = set(scan.observed_paths)
        issues = list(scan.issues)
        for scanned in scan.files:
            if (
                update
                and self._repository.file_hash(repository_id, scanned.relative_path)
                == scanned.sha256
            ):
                unchanged += 1
                continue
            try:
                functions = self._parser.extract(scanned.source, scanned.relative_path)
            except SyntaxError:
                issues.append(ScanIssue(scanned.relative_path, "Python syntax could not be parsed"))
                self._repository.record_file_error(
                    repository_id, scanned.relative_path, scanned.sha256
                )
                continue
            self._repository.replace_file_functions(
                repository_id, scanned.relative_path, scanned.sha256, functions
            )
            indexed += 1
            function_count += len(functions)
        removed = 0
        removed = self._repository.remove_unseen_files(repository_id, seen_paths)
        return IndexSummary(
            repository_id=repository_id,
            discovered_files=len(scan.files),
            indexed_files=indexed,
            unchanged_files=unchanged,
            removed_files=removed,
            function_count=function_count,
            issues=tuple(issues),
            complete=True,
        )
