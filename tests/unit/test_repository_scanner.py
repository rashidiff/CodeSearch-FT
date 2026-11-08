"""Safety tests for bounded source discovery."""

from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, cast

from pytest import MonkeyPatch

from codesearch_ft.ingestion.repository_scanner import IndexingOptions, RepositoryScanner


def test_scanner_respects_gitignore_exclusions_and_utf8(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\ncache/\n", encoding="utf-8")
    (tmp_path / "kept.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def ignored(): pass\n", encoding="utf-8")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "cached.py").write_text("def cached(): pass\n", encoding="utf-8")
    (tmp_path / "custom.py").write_text("def custom(): pass\n", encoding="utf-8")

    result = RepositoryScanner(IndexingOptions(excluded_patterns=("custom.py",))).scan(tmp_path)

    assert [file.relative_path for file in result.files] == ["kept.py"]
    assert result.complete is True


def test_scanner_skips_binary_invalid_utf8_and_oversized_files(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"\x00")
    (tmp_path / "encoding.py").write_bytes(b"\xff")
    (tmp_path / "large.py").write_text("x" * 20, encoding="utf-8")

    result = RepositoryScanner(IndexingOptions(max_file_bytes=10)).scan(tmp_path)

    assert not result.files
    assert {issue.reason for issue in result.issues} == {
        "binary content is not indexed",
        "file is not valid UTF-8",
        "file exceeds configured size limit",
    }


def test_scanner_respects_python_encoding_cookie(tmp_path: Path) -> None:
    source = "# coding: latin-1\ndef café(): return 'olé'\n".encode("latin-1")
    (tmp_path / "latin.py").write_bytes(source)

    result = RepositoryScanner().scan(tmp_path)

    assert len(result.files) == 1
    assert "café" in result.files[0].source


def test_scanner_does_not_read_a_path_resolving_outside_root(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    external = root / "external.py"
    external.write_text("def secret(): pass\n", encoding="utf-8")
    outside_file = outside / "secret.py"
    outside_file.write_text("def secret(): pass\n", encoding="utf-8")
    original_resolve = Path.resolve
    original_open: Callable[..., IO[Any]] = Path.open

    def resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path == external:
            return outside_file
        return original_resolve(path, *args, **kwargs)

    def open_file(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == external:
            raise AssertionError("outside-root candidate was read")
        return cast(IO[Any], original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "open", open_file)

    result = RepositoryScanner().scan(root)

    assert not result.files
    assert result.complete is False
    assert result.issues[0].reason == "file could not be read within repository root"


def test_file_limit_marks_scan_incomplete(tmp_path: Path) -> None:
    for filename in ("a.py", "b.py"):
        (tmp_path / filename).write_text("def f(): pass\n", encoding="utf-8")

    result = RepositoryScanner(IndexingOptions(max_files=1)).scan(tmp_path)

    assert result.complete is False
    assert result.issues[-1].reason == "repository file limit exceeded"
