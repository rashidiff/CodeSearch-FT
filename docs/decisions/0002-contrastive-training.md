# ADR 0002: In-batch contrastive fine-tuning

- **Status:** Accepted by approved project design
- **Context:** Training pairs associate natural-language descriptions with Python functions and the target is retrieval ranking.
- **Decision:** Optimize temperature-scaled diagonal-positive cross-entropy over a batch similarity matrix using actual PyTorch gradients; make symmetric loss optional and guard against duplicate false negatives.
- **Alternatives:** Classification over a fixed label set does not match open-corpus retrieval; frozen embeddings do not demonstrate fine-tuning; generated negatives risk adding unverified labels.
- **Consequences:** Batches provide negatives efficiently but may contain equivalent examples. Data identity filtering, seed/config recording, and CPU synthetic gradient tests are required. No training occurs in Phase 1.
