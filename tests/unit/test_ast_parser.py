"""Unit tests for safe AST extraction and source-preserving metadata."""

from pathlib import Path

import pytest

from codesearch_ft.ingestion.ast_parser import AstCodeParser


def test_extracts_module_class_async_and_nested_functions_with_qualified_names() -> None:
    source = """
def outer(item: str, /, keep: bool = True) -> list[str]:
    def inner(value: str) -> str:
        return value.strip()
    return [inner(item)] if keep else []

class Worker:
    async def run(self, value: str) -> str:
        async def normalize() -> str:
            return value.strip()
        return await normalize()
"""

    functions = AstCodeParser().extract(source, "worker.py")

    assert [function.qualified_name for function in functions] == [
        "outer",
        "outer.<locals>.inner",
        "Worker.run",
        "Worker.run.<locals>.normalize",
    ]
    assert functions[0].signature == "def outer(item: str, /, keep: bool=True) -> list[str]"
    assert functions[2].signature.startswith("async def run(self, value: str)")
    assert all(function.file_path == "worker.py" for function in functions)


def test_preserves_decorators_and_line_spans_but_strips_top_level_docstring() -> None:
    source = '''
@staticmethod
def load(path: str):
    """Read the data and return a mapping."""
    def parse() -> dict[str, str]:
        "Nested helper documentation remains in the outer implementation input."
        return {}
    return parse()
'''

    function = AstCodeParser().extract(source, "loader.py")[0]

    assert function.line_start == 2
    assert function.line_end == 8
    assert function.source_code.startswith("@staticmethod\ndef load")
    assert function.docstring == "Read the data and return a mapping."
    assert "Read the data and return a mapping" not in function.embedding_input
    assert "Nested helper documentation remains" in function.embedding_input
    assert function.content_sha256


def test_repeated_qualified_names_get_deterministic_symbol_keys() -> None:
    source = "def item():\n    return 1\n\ndef item():\n    return 2\n"

    functions = AstCodeParser().extract(source, "duplicates.py")

    assert [function.qualified_name for function in functions] == ["item", "item"]
    assert [function.symbol_key for function in functions] == ["item", "item#duplicate-2"]


def test_syntax_error_is_reported_and_source_is_never_executed(tmp_path: Path) -> None:
    parser = AstCodeParser()
    side_effect = tmp_path / "executed.txt"
    source = (
        f"from pathlib import Path\nPath({str(side_effect)!r}).write_text('executed')\n"
        "def broken(:\n    pass\n"
    )

    with pytest.raises(SyntaxError):
        parser.extract(source, "untrusted.py")

    assert not side_effect.exists()


def test_empty_source_produces_no_functions() -> None:
    assert AstCodeParser().extract("", "empty.py") == ()
