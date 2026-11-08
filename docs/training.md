# Fine-tuning

CodeSearch-FT uses the pinned `microsoft/codebert-base` encoder and a PyTorch/Accelerate in-batch InfoNCE trainer. The model revision, dataset revision, seed, effective configuration, split counts, duplicate policy and hashes are recorded with each run.

## Data and leakage controls

The adapter reads bounded rows from the pinned `code-search-net/code_search_net` Python configuration and preserves the official train, validation and test partitions. It removes the target function docstring from code input, detects exact duplicate query/code identities, quarantines groups that cross partitions, reports near-duplicate candidates and repository overlap when metadata permits, and writes a hashed dataset manifest. The source declares `license: other` and does not provide per-example license metadata; check upstream repository terms before redistributing source examples.

CodeBERT pretraining overlaps CodeSearchNet. Training and benchmark results therefore do not establish generalization to unseen source repositories.

## Google Colab pilot

Open [`notebooks/codesearch_ft_colab_pilot.ipynb`](../notebooks/codesearch_ft_colab_pilot.ipynb) in Google Colab. Select a GPU runtime, upload the generated deployment bundle when prompted, and follow the preflight before starting the explicitly bounded GPU run. The notebook uses the packaged training implementation, real CodeSearchNet Python train/validation data and a pinned CodeBERT revision. It does not run training merely by opening the notebook.

Mount Google Drive from the notebook to persist checkpoints across runtime loss. The notebook saves the Transformers model/tokenizer export, optimizer/resume state, logs, metrics and data manifest under the configured Drive output directory. Review Drive permissions in Colab before mounting.

The pilot profile and notebook enforce a bounded sample suitable for one T4. Results depend on the runtime image and GPU availability.

## Local training

Install the training group and run an explicit profile:

```bash
uv sync --locked --extra inference --group training
uv run codesearch config validate configs/train_cpu_smoke.yaml --kind training
uv run codesearch train --config configs/train_cpu_smoke.yaml
```

The CPU smoke profile validates the real trainer with tiny local fixtures/configuration. Larger profiles can download model and dataset dependencies and require explicit resource planning. Checkpoint output includes exportable model files, run metadata, optimizer/scheduler progress and RNG state when supported.
