# ADR 0006: Exact FAISS before approximate search

- **Status:** Accepted by approved project design
- **Context:** Retrieval quality and correctness must be measurable before optimizing large-corpus search speed.
- **Decision:** Begin with normalized vectors in exact FAISS `IndexFlatIP`; evaluate approximate indexing only after profiling establishes a need and quality trade-offs are measured.
- **Alternatives:** Approximate indexes introduce tuning and recall loss before scale requirements are known.
- **Consequences:** Initial vector search is transparent and exact but linear in corpus size. Index size and latency are benchmarked before considering a change. FAISS integration is deferred to Phase 3.
