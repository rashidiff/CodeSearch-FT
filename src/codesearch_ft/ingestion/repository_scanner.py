"""Bounded, non-executing traversal of local Python repositories."""

import fnmatch
import hashlib
import io
import os
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from codesearch_ft.domain.errors import IndexingError


@dataclass(frozen=True, slots=True)
class IndexingOptions:
    """Conservative limits and explicit file exclusions for source scanning."""

    max_file_bytes: int = 1_000_000
    max_files: int = 100_000
    excluded_dirs: tuple[str, ...] = (".git", ".venv", "venv", "__pycache__", "node_modules")
    excluded_patterns: tuple[str, ...] = ("**/build/**", "**/dist/**", "**/*.generated.py")

    def __post_init__(self) -> None:
        if self.max_file_bytes < 1 or self.max_files < 1:
            raise ValueError("Indexing limits must be positive integers.")


@dataclass(frozen=True, slots=True)
class ScannedFile:
    """Successfully decoded Python source and byte-level identity."""

    relative_path: str
    source: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ScanIssue:
    """A path and concise reason for skipping it without logging file contents."""

    relative_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    """A complete or incomplete repository traversal result."""

    files: tuple[ScannedFile, ...]
    issues: tuple[ScanIssue, ...]
    observed_paths: tuple[str, ...]
    complete: bool


class RepositoryScanner:
    """Walk a resolved root, never following symlinks outside it."""

    def __init__(self, options: IndexingOptions | None = None) -> None:
        self.options = options or IndexingOptions()

    def scan(self, root: Path) -> ScanResult:
        try:
            resolved_root = root.expanduser().resolve(strict=True)
        except OSError as exc:
            raise IndexingError(f"Repository path is unavailable: {root}") from exc
        if not resolved_root.is_dir():
            raise IndexingError(f"Repository path is not a directory: {resolved_root}")

        ignore_rules = _read_gitignore(resolved_root)
        discovered: list[ScannedFile] = []
        issues: list[ScanIssue] = []
        observed: list[str] = []
        complete = True
        file_count = 0

        for directory, child_dirs, filenames in os.walk(resolved_root, followlinks=False):
            directory_path = Path(directory)
            relative_dir = directory_path.relative_to(resolved_root).as_posix()
            retained_dirs: list[str] = []
            for child in child_dirs:
                child_path = directory_path / child
                relative_child = _join_relative(relative_dir, child)
                if child_path.is_symlink():
                    issues.append(
                        ScanIssue(relative_child, "symbolic directory link is not followed")
                    )
                    continue
                if (
                    child in self.options.excluded_dirs
                    or _is_ignored(relative_child, True, ignore_rules)
                    or _matches(relative_child, self.options.excluded_patterns)
                ):
                    continue
                retained_dirs.append(child)
            child_dirs[:] = retained_dirs
            for filename in sorted(filenames):
                candidate = directory_path / filename
                relative_path = candidate.relative_to(resolved_root).as_posix()
                if not filename.endswith(".py"):
                    continue
                if candidate.is_symlink():
                    issues.append(ScanIssue(relative_path, "symbolic file link is not followed"))
                    continue
                if _is_ignored(relative_path, False, ignore_rules) or _matches(
                    relative_path, self.options.excluded_patterns
                ):
                    continue
                observed.append(relative_path)
                file_count += 1
                if file_count > self.options.max_files:
                    issues.append(ScanIssue(relative_path, "repository file limit exceeded"))
                    complete = False
                    break
                try:
                    resolved_file = candidate.resolve(strict=True)
                    resolved_file.relative_to(resolved_root)
                    stat = resolved_file.stat()
                    if stat.st_size > self.options.max_file_bytes:
                        issues.append(
                            ScanIssue(relative_path, "file exceeds configured size limit")
                        )
                        continue
                    with resolved_file.open("rb") as source_file:
                        payload = source_file.read(self.options.max_file_bytes + 1)
                    if len(payload) > self.options.max_file_bytes:
                        issues.append(
                            ScanIssue(relative_path, "file exceeds configured size limit")
                        )
                        continue
                    if b"\x00" in payload:
                        issues.append(ScanIssue(relative_path, "binary content is not indexed"))
                        continue
                    encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
                    source = payload.decode(encoding)
                except (UnicodeDecodeError, LookupError, SyntaxError):
                    issues.append(ScanIssue(relative_path, "file is not valid UTF-8"))
                    continue
                except (OSError, ValueError):
                    issues.append(
                        ScanIssue(relative_path, "file could not be read within repository root")
                    )
                    complete = False
                    continue
                discovered.append(
                    ScannedFile(relative_path, source, hashlib.sha256(payload).hexdigest())
                )
            if not complete:
                break

        return ScanResult(tuple(discovered), tuple(issues), tuple(observed), complete)


def _read_gitignore(root: Path) -> tuple[str, ...]:
    ignore_file = root / ".gitignore"
    try:
        lines = ignore_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeDecodeError):
        return ()
    return tuple(
        line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")
    )


def _is_ignored(path: str, is_dir: bool, rules: Iterable[str]) -> bool:
    normalized = path.strip("/")
    ignored = False
    for raw_rule in rules:
        negated = raw_rule.startswith("!")
        rule = raw_rule[1:] if negated else raw_rule
        directory_rule = rule.endswith("/")
        rule = rule.strip("/")
        if not rule:
            continue
        has_slash = "/" in rule
        candidates = [normalized] if has_slash else normalized.split("/")
        matches = any(fnmatch.fnmatchcase(item, rule) for item in candidates)
        if matches and (not directory_rule or is_dir or "/" in normalized):
            ignored = not negated
    return ignored


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _join_relative(parent: str, child: str) -> str:
    return f"{parent}/{child}" if parent else child
