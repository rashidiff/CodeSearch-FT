# Dependency compatibility decision

**Checked:** 2026-11-05

This note records compatibility facts used for Phase 1 dependency ranges. It is not a substitute for the resolved `uv.lock`; the lock is the reproducible selected graph. The project target remains CPython 3.11 and 3.12. WSL2 Ubuntu x86-64 is the reference development environment; native Windows support must be verified by actual lock/install/CI checks before being promised.

## Verified from primary sources

- uv lists CPython 3.11 and 3.12 as Tier 1 supported versions and supports project dependency groups and lock/sync workflows: <https://docs.astral.sh/uv/reference/policies/python/>, <https://docs.astral.sh/uv/concepts/dependencies/>, <https://docs.astral.sh/uv/concepts/projects/sync/>.
- PyTorch's installation guidance says latest PyTorch requires Python 3.9+ and recommends Python 3.9–3.12 for its macOS instructions; its installer explicitly offers CPU builds: <https://pytorch.org/get-started/locally/>. Python 3.11/3.12 are within that stated range.
- Hugging Face Transformers installation documentation says it is tested with Python 3.11+ and PyTorch 2.6+ and documents CPU installation: <https://huggingface.co/docs/transformers/installation>.
- Accelerate documents testing on Python 3.11+: <https://huggingface.co/docs/accelerate/main/basic_tutorials/install>.
- Datasets documents testing on Python 3.10+: <https://huggingface.co/docs/datasets/installation>.
- MLflow's Python API and local tracking documentation expose the selected tracking surface; the resolved Python compatibility is additionally checked by uv's lock resolution: <https://mlflow.org/docs/latest/api_reference/python_api/index.html>, <https://mlflow.org/docs/latest/ml/tracking/>.
- FastAPI's current first-steps documentation demonstrates Python 3.10+ and automatic OpenAPI generation; its migration documentation records Pydantic v2 support from FastAPI 0.100.0: <https://fastapi.tiangolo.com/tutorial/first-steps/>, <https://fastapi.tiangolo.com/how-to/migrate-from-pydantic-v1-to-pydantic-v2/>.
- Current Pydantic package metadata describes the v2 line as Python 3.10+, and its project page identifies v2: <https://pypi.org/project/pydantic/>.
- Typer release notes document Python 3.8+ for releases after 0.19.1; the latest package page and docs provide the CLI interface used here: <https://typer.tiangolo.com/release-notes/>.
- FAISS's own install guide recommends Conda for supported packages, while its current CPU-only package availability statement includes Windows x86-64 and Linux x86-64/aarch64; PyPI separately publishes `faiss-cpu`: <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>, <https://pypi.org/project/faiss-cpu/>. This does **not** establish that a compatible PyPI wheel exists for every Python/OS/architecture combination in this project.
- Ruff uses `project.requires-python` to infer a compatible target version when `target-version` is omitted: <https://docs.astral.sh/ruff/configuration/>.

## Selected policy for Phase 1

- Declare `requires-python = ">=3.11,<3.13"`.
- Use bounded compatible-major ranges for direct dependencies and let uv produce exact transitive/direct selections in `uv.lock`.
- Keep lightweight API/CLI/configuration dependencies in the default install. Keep CodeBERT inference and training stacks in named optional dependency groups so a basic CPU developer install does not automatically fetch heavyweight model libraries.
- Resolve every declared group for both Python 3.11 and 3.12 in the lock. Run the actual test/quality environment on the locally available Python 3.11.9. Native Windows and Linux wheel installation checks for FAISS/PyTorch are separate from successful metadata resolution.
- Do not download model weights or datasets in Phase 1.

## Resolved versions and local evidence

On 2026-11-05, uv 0.9.26 resolved 159 packages in the universal lock. The direct graph selected FastAPI 0.162.0, Pydantic 2.14.3, PyYAML 6.0.3, Typer 0.27.3, Uvicorn 0.54.0, PyTorch 2.15.0, Transformers 5.22.1, FAISS CPU 1.15.1, Accelerate 1.15.0, Datasets 4.8.5, MLflow 3.18.0, Pytest 8.4.2, pytest-cov 6.3.0, Ruff 0.17.3, and Pyright 1.1.414. `uv lock --check` passed. The lightweight development install and full quality checks passed on Python 3.12.12 and Python 3.11.9. Inference/training wheels were resolved into the lock but intentionally not installed in Phase 1; their binary availability is not claimed by this note.

## Compatibility limits

The references establish advertised Python ranges and supported installation paths, not the compatibility of every latest package combination. The actual `uv lock` result and offline Phase 1 checks provide the local evidence. FAISS wheel availability and binary loading must be exercised in Phase 3 on each claimed platform. Dataset loading APIs, CodeBERT revision, and CodeSearchNet schema/provenance are not part of this Phase 1 verification.
