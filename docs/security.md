# Security review

This project indexes source as untrusted text. It does not import or execute repository modules. The filesystem scanner resolves the selected root, applies traversal bounds and exclusions, skips symlinks, limits file sizes, and refuses incomplete scans before pruning metadata.

Database access uses SQLite parameter binding for values. FTS input is reduced to quoted terms instead of passing user operators through. The API defaults to `127.0.0.1`, validates bounded query and top-k fields, rejects extra request fields, exposes no path-indexing or training route, and omits local artifact paths from responses. Search errors are returned as bounded structured messages; unexpected adapter errors are logged without source snippets.

Model and vector artifacts are checked against version, dimensions, mapping and SHA-256 metadata before use. Keep local SQLite databases protected because they contain indexed source snippets. Do not expose the API to an untrusted network; binding beyond localhost is an explicit operator choice.

## Review record

- Filesystem traversal: bounded and root-contained; symlinks are skipped. Incomplete scans do not delete unseen records.
- Source/log disclosure: source excerpts are returned only to the local search caller; scanner and error logs do not include snippets.
- SQL: data values use parameterized statements; query syntax is normalized before FTS execution.
- Artifact loading: model revision and artifact integrity checks are enforced for semantic indexes.
- API: query length and result limits are validated; no indexing/training endpoints exist.
- CI: workflow uses read-only repository contents permission and runs deterministic checks without training or data downloads.

No unresolved high-risk issue was found in this source review. Local database contents, model cache files and any explicitly exposed API remain the operator's responsibility.
