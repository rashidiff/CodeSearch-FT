# ADR 0005: CPU-first local inference

- **Status:** Accepted by approved project design
- **Context:** Local development has no dedicated GPU and core product operation must remain available without CUDA.
- **Decision:** Make the base CLI/API and tests CPU-capable; treat model inference as an optional local dependency path and GPU use as an explicit training profile.
- **Alternatives:** Requiring CUDA would exclude the normal development machine; hosted inference would require network access, credentials, or paid services.
- **Consequences:** Real neural inference needs local model weights and may be slow on CPU. Missing weights must be reported, never hidden by dummy embeddings. Phase 1 health/configuration features do not load weights.
