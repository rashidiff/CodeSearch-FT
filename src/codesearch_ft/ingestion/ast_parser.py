"""Safe extraction of Python function definitions from source text."""

import ast
import copy
import hashlib
from collections import defaultdict
from collections.abc import Iterable

from codesearch_ft.domain.models import ExtractedFunction


class AstCodeParser:
    """Extract functions without importing, compiling, or executing repository code."""

    def extract(self, source: str, relative_path: str) -> tuple[ExtractedFunction, ...]:
        """Parse source and return functions in source order.

        Repeated definitions with the same qualified name receive an ordinal suffix in
        ``symbol_key``. That key stays stable when a function body changes but the
        definition order in its lexical scope does not.
        """
        tree = ast.parse(source, filename=relative_path, type_comments=True)
        visitor = _FunctionVisitor(source, relative_path)
        visitor.visit(tree)
        return tuple(visitor.functions)


class _FunctionVisitor(ast.NodeVisitor):
    def __init__(self, source: str, relative_path: str) -> None:
        self._lines = source.splitlines(keepends=True)
        self._relative_path = relative_path
        self._scope: list[str] = []
        self._scope_kinds: list[str] = []
        self._occurrences: defaultdict[str, int] = defaultdict(int)
        self.functions: list[ExtractedFunction] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node.name, "class", node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.functions.append(self._extract_function(node))
        self._visit_scope(node.name, "function", node.body)

    def _visit_scope(self, name: str, kind: str, body: Iterable[ast.stmt]) -> None:
        previous_scope = self._scope
        previous_kinds = self._scope_kinds
        next_scope = [*previous_scope]
        if previous_kinds and previous_kinds[-1] == "function":
            next_scope.append("<locals>")
        next_scope.append(name)
        self._scope = next_scope
        self._scope_kinds = [*previous_kinds, kind]
        try:
            for statement in body:
                self.visit(statement)
        finally:
            self._scope = previous_scope
            self._scope_kinds = previous_kinds

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ExtractedFunction:
        qualified_parts = [*self._scope]
        if self._scope_kinds and self._scope_kinds[-1] == "function":
            qualified_parts.append("<locals>")
        qualified_name = ".".join([*qualified_parts, node.name])
        occurrence = self._occurrences[qualified_name]
        self._occurrences[qualified_name] += 1
        symbol_key = (
            qualified_name if occurrence == 0 else f"{qualified_name}#duplicate-{occurrence + 1}"
        )

        start_line = min(
            (decorator.lineno for decorator in node.decorator_list),
            default=node.lineno,
        )
        end_line = node.end_lineno or node.lineno
        source_code = "".join(self._lines[start_line - 1 : end_line])
        if not source_code:
            source_code = ast.get_source_segment("".join(self._lines), node) or ""

        docstring = ast.get_docstring(node, clean=False)
        embedding_input = _code_without_own_docstring(node)
        return_annotation = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        signature = f"{async_prefix}def {node.name}({ast.unparse(node.args)}){return_annotation}"
        digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()

        return ExtractedFunction(
            symbol_key=symbol_key,
            name=node.name,
            qualified_name=qualified_name,
            signature=signature,
            file_path=self._relative_path,
            line_start=start_line,
            line_end=end_line,
            source_code=source_code,
            embedding_input=embedding_input,
            content_sha256=digest,
            docstring=docstring,
        )


def _code_without_own_docstring(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    body = list(node.body)
    if body and _is_docstring(body[0]):
        body.pop(0)
    clean_node = copy.copy(node)
    clean_node.body = body or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(clean_node))


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )
