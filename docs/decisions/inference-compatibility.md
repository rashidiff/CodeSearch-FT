# Phase 3 inference compatibility record

Verified 2026-11-05 for this workspace's Windows 11 x86-64, CPython 3.12.12
environment. The user authorized downloading only the pinned CodeBERT checkpoint
for the real CPU smoke test; no dataset or training was run.

## CodeBERT

- Model: `microsoft/codebert-base`, model revision
  `3b0952feddeffad0063f274080e3c23d75e7eb39` (immutable Hub commit).
- The Hub model metadata identifies `AutoTokenizer` and `AutoModel` as its
  Transformers interfaces. Its checked-in `config.json` identifies a
  `RobertaModel` with hidden size 768, 12 layers, and 12 attention heads.
- The loader pins that revision for both tokenizer and weights and defaults to
  `local_files_only=True`. Network access is possible only through the explicit
  `codesearch models download` command or an explicit programmatic opt-in.
- The PyTorch checkpoint listed by the model repository is about 499 MB. Budget
  at least 2 GB of free RAM for a small CPU inference process as a conservative
  planning estimate. Actual peak memory and throughput have not been measured.
  `codesearch models download` loaded the pinned weights successfully, and
  `codesearch models smoke` completed real CPU query and code embeddings with
  768 dimensions. The default device is CPU and does not probe or require CUDA.
- Locked/runtime versions and local import evidence: PyTorch 2.15.0+cpu,
  Transformers 5.22.1, NumPy 2.5.4. Real checkpoint loading was verified by the
  smoke command. Default automated tests continue to use synthetic model and
  tokenizer fixtures and require no cached weights.

## FAISS CPU

- `faiss-cpu==1.15.1` provides a CPython 3.12 Windows x86-64 wheel. It was
  installed into the project's existing virtual environment and imported on
  the actual Windows host. A real `IndexIDMap2(IndexFlatIP)` was created, filled
  with normalized synthetic vectors, and returned the expected exact ranking.
- The checked runtime reported FAISS 1.15.1 and NumPy 2.5.4. Phase 3 tests also
  exercise actual FAISS save/load and ID mappings. Native Windows CPU FAISS is
  therefore verified for this x86-64/Python 3.12 environment. This does not
  establish support for every Python/platform combination; WSL2 Ubuntu remains
  a documented alternative if a future wheel is unavailable.

## References

- [CodeBERT model metadata and immutable revision](https://huggingface.co/api/models/microsoft/codebert-base)
- [CodeBERT model configuration](https://huggingface.co/microsoft/codebert-base/blob/main/config.json)
- [Transformers Auto classes](https://huggingface.co/docs/transformers/model_doc/auto)
- [Transformers model loading and revision API](https://huggingface.co/docs/transformers/main_classes/model)
- [FAISS installation and supported CPU platforms](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
- [FAISS CPU wheel files](https://pypi.org/project/faiss-cpu/1.15.1/)
