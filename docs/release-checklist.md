# Release checklist

## Contents

- [x] Source package, configs and deterministic test suite included.
- [x] Locked Python dependencies and CI quality workflow and separate manual Docker smoke workflow included.
- [x] CPU-only container definition and a manually triggered Docker smoke workflow for health and lexical search included; the local image build and container smoke passed.
- [x] Both Colab notebooks included without saved execution output.
- [x] Compact benchmark metrics, uncertainty estimates, fixed-set manifest and verification record included.
- [x] MIT license included, consistent with package metadata.
- [x] No dataset rows, checkpoint weights, model cache, database, vector index or credentials are part of the public tree.
- [x] Recursive audit found zero disallowed names or text.

## Reproducibility and limitations

The recorded benchmark uses automatic relevance associations, 124 queries and 128 candidates. See `evaluation.md` for dataset-overlap and generalization limitations. The CPU profile in `performance.md` is a generated fixture, not an external benchmark. CI covers Python 3.11 and 3.12.

The full offline-flagged test suites and isolated package builds pass in both the development and public directories. The public image built and passed container health, indexing and CLI/API lexical search with a mounted local fixture. Pinned CodeBERT weights are not cached for the optional model smoke. See `verification.md` for exact results and warnings.
