# Semantic and hybrid retrieval (Phase 3)

## Encoder

`CodeBERTEncoder` uses `AutoTokenizer` and `AutoModel` from Transformers for
the pinned `microsoft/codebert-base` commit recorded in
[`inference-compatibility.md`](../decisions/inference-compatibility.md).
Query and docstring-free function implementation are encoded independently
through the same model. The model's last hidden states are attention-mask-aware
mean pooled and L2 normalized, so inner product is cosine similarity. Maximum
sequence lengths and embedding batches are configurable in the encoder. CPU is
the default and missing local weights produce an actionable exception; no
inference path silently switches to a synthetic encoder. To fetch the pinned
weights, run `codesearch models download` explicitly. That command requires
network access and substantial disk/RAM resources.

## FAISS artifact and consistency

Each semantic build creates a new `IndexIDMap2(IndexFlatIP)` using explicit
SQLite function IDs as vector IDs. Embeddings are produced in batches; FAISS
stores normalized float32 vectors. A JSON sidecar records manifest version,
repository, model/revision, preprocessing version, dimension, count, corpus
fingerprint, and SHA-256 of the FAISS artifact. SQLite `index_versions` stores
the same compatibility and integrity fields, while `vector_mapping` provides
the relational function-to-vector mapping.

Builds are written to unique staged files under the database's
`semantic_indexes/` directory. After the FAISS artifact is flushed and hashed,
the corpus snapshot and mappings are validated. A SQLite transaction then
promotes the new version and marks the prior ready version stale. Files are
renamed to their immutable final version before promotion; failures retain the
previous ready version and remove the staged version/artifacts. Readers use
only `ready` versions and validate sidecar/SQLite metadata, artifact hash,
model revision, preprocessing version, dimension, exact FAISS type, count, and
function IDs before querying. A source update marks the active semantic index
stale; explicitly rebuild with:

```powershell
codesearch index .\my-python-project --update --semantic
```

If files are changed concurrently during indexing, the corpus fingerprint
check rejects promotion instead of activating a partial or mismatched index.
An unavailable or corrupt active artifact is an error; it is never replaced
with fake vectors. Recovery is to rebuild the semantic index.

## Search modes

- `lexical`: SQLite FTS5 BM25 only; lower BM25 score is better.
- `semantic`: exact FAISS inner-product similarity over normalized vectors;
  larger cosine similarity is better. This score is not a probability.
- `hybrid`: takes lexical and semantic ranked lists and applies Reciprocal Rank
  Fusion. With one-based rank `r` and configurable constant `k`, a result gets
  `sum(1 / (k + r))` across lists in which it appears. Final ties use stable
  function IDs. API/CLI results label each score's semantics.

The local search API (`POST /v1/search`) accepts `query`, `mode`, bounded
`top_k`, optional `index_id`, and repository filter. It does not accept paths,
index repositories, or launch training. The CLI serves on localhost by default.
The model/index registry (`codesearch models list`, `codesearch indexes list`,
and `GET /v1/models`, `GET /v1/indexes`) reports identifiers and state without
loading large model weights.

## Limitations

The FAISS reference index is exact and intended to establish correctness, not
unbounded scale. Semantic CPU latency and memory have not been benchmarked.
The model revision, library compatibility, and native Windows FAISS wheel were
verified, but actual checkpoint loading/inference is pending explicit weight
download authorization. Offline automated tests exercise PyTorch pooling and
the full retrieval/index path with deterministic local fixtures and actual
FAISS CPU; they do not claim CodeBERT retrieval quality.
