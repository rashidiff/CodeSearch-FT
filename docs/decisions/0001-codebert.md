# ADR 0001: CodeBERT as the shared retrieval encoder

- **Status:** Accepted by approved project design
- **Context:** Search must compare natural-language intent with Python implementations, and the project must demonstrate code-model fine-tuning.
- **Decision:** Use `microsoft/codebert-base` through Hugging Face `AutoTokenizer` and `AutoModel`, with one shared backbone for query and code inputs.
- **Alternatives:** General-purpose embedding APIs were excluded because they add external service/key requirements and do not provide the required PyTorch fine-tuning. Generic text encoders were not selected because the project specifically targets code representation.
- **Consequences:** Inference is local but model downloads are required when weights are not cached; CodeBERT-related pretraining data overlap must be disclosed in evaluation. Phase 1 records this choice but does not load or download weights.
