# ADR 0003: SQLite metadata with FAISS CPU vectors

- **Status:** Accepted by approved project design
- **Context:** The initial product is a local single-user tool that needs both inspectable source metadata/lexical retrieval and dense vector similarity.
- **Decision:** Store relational metadata and FTS5 lexical records in SQLite; persist normalized vectors in a versioned FAISS CPU `IndexFlatIP` artifact with explicit vector-to-function mappings in SQLite.
- **Alternatives:** A remote database/service adds operational requirements; putting large vector arrays in SQLite or JSON conflates storage roles; approximate FAISS indexing is unnecessary before measured scale requires it.
- **Consequences:** SQLite/FAISS promotion requires staged artifacts, metadata validation, and rollback because they cannot share one transaction. Exact search provides a trusted baseline. Neither index is implemented in Phase 1.
