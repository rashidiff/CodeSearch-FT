# Local indexing and lexical retrieval (Phase 2)

Phase 2 indexes Python source as untrusted text. `RepositoryScanner` walks only
the selected root, skips symbolic links and common generated/environment
directories, reads UTF-8 Python files up to 1,000,000 bytes, and stops after
100,000 eligible files by default. CLI `--max-file-bytes`, `--max-files`, and
repeatable `--exclude` options can lower or raise those bounds. Root `.gitignore`
rules are honored on a best-effort basis; nested `.gitignore` files and the
complete Git ignore specification are not implemented. Invalid encoding,
binary, oversized, and invalid-syntax files are reported by path and reason;
source contents are not logged. Directory symlinks are pruned and file symlinks
are skipped. No repository file is imported or executed.

`AstCodeParser` extracts module functions, class methods, async functions, and
nested definitions. It retains original snippets, decorator-inclusive source
line starts, end lines, signatures, optional docstrings, and SHA-256 content
identity. Its separate `embedding_input` removes that function's own leading
docstring. Repeated qualified names
use an ordinal suffix in `symbol_key` to keep database records distinct.

## SQLite schema and updates

The default database is the platform user-data path returned by Typer's
`get_app_dir("codesearch-ft")`, ending in `metadata.sqlite3`. Override it with
`--database PATH` or `CODESEARCH_DATABASE`. This avoids writing artifacts into
indexed repositories. The database is local and may contain source snippets;
protect it like the repositories it indexes. `.gitignore` excludes database and
`.codesearch` artifacts.

Numbered migrations are tracked in `schema_migrations`. Migration 1 creates:

- `repositories`, unique by canonical root path;
- `source_files`, unique by repository and relative path, with byte hash/status;
- `functions`, unique by source file and stable symbol key, with original and
  docstring-free code forms and checked source line ranges;
- `index_versions`, for future model/preprocessing/dimension-bound vector builds;
- `vector_mapping`, uniquely mapping function/version and vector IDs;
- `function_fts`, an external-content FTS5 index over name, qualified name,
  signature, docstring, and source code.

Foreign keys cascade on deletion. Insert/update/delete triggers synchronize FTS
inside the same SQLite transaction as metadata. Each changed file is parsed
before its transactional replacement. Unchanged bytes are skipped with
`--update`; absent files are removed after a complete scan. If traversal is
incomplete, the indexing run fails before changing metadata, preventing a
partial scan from deleting unseen records. Files that are still present but
cannot be read remain known and do not cause existing metadata to be pruned.

FTS query input is split into alphanumeric/underscore terms and quoted as
individual FTS tokens joined by `OR`; operators and punctuation are never passed
through as query syntax. Empty/punctuation-only input returns no results. The
SQLite query is parameterized, ranks by `bm25(function_fts, 3, 2, 1.5, 1, 1)`
ascending, then by stable integer function ID. The score is explicitly labeled
`sqlite_fts5_bm25 (lower is better)` and is not a probability.

## CLI

```powershell
codesearch index .\my-python-project --database .\local-index.sqlite3
codesearch index .\my-python-project --update --database .\local-index.sqlite3
codesearch search "dict fromkeys" --mode lexical --top-k 10 --database .\local-index.sqlite3
```

Lexical, semantic, and hybrid retrieval are available after Phase 3. Semantic
index generation and compatibility checks are described in
[`semantic-retrieval.md`](semantic-retrieval.md). The API exposes search but
does not expose filesystem indexing over HTTP.
