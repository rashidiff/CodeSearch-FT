"""Reproducible Accelerate trainer and CodeBERT-compatible checkpoint workflow."""

# Optional ML framework APIs have incomplete third-party stubs; public inputs/outputs remain typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from codesearch_ft.config.models import TrainingConfig
from codesearch_ft.datasets.codesearchnet import (
    DatasetManifest,
    TrainingPair,
    iter_source_splits,
    prepare_partitions,
    write_manifest,
)
from codesearch_ft.ml.contrastive import info_nce_loss

CHECKPOINT_FORMAT = 1


def train(
    config: TrainingConfig,
    *,
    pairs: dict[str, list[TrainingPair]] | None = None,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Run configured training, validation, checkpointing, and local MLflow tracking."""
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(config.random_seed)
    random.seed(config.random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if pairs is None and resume is not None:
        pairs = _read_manifest_pairs(config.output_dir / "dataset-manifest.json")
        if pairs is None:
            raise ValueError("Resume requires sampled training-pairs.json in the output directory.")
    if pairs is None:
        raw = iter_source_splits(
            config,
            {
                "train": config.max_train_samples,
                "validation": config.validation_max_samples,
                "test": config.test_max_samples,
            },
        )
    else:
        raw = pairs
    partitions, manifest = prepare_partitions(raw, config)
    if not partitions["train"]:
        raise ValueError("No usable training pairs remain after preprocessing and leakage checks.")
    if len(partitions["train"]) < 2:
        raise ValueError("InfoNCE training needs at least two unique training pairs per batch.")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=_precision(config.mixed_precision, torch),
    )
    if accelerator.num_processes > 1:
        set_seed(config.random_seed, device_specific=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.model_revision, local_files_only=resume is not None
    )
    if resume is None:
        model = AutoModel.from_pretrained(
            config.base_model, revision=config.model_revision, local_files_only=False
        )
    else:
        metadata = json.loads((resume / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata.get("model_id") != config.base_model
            or metadata.get("model_revision") != config.model_revision
        ):
            raise ValueError(
                "Resume checkpoint base model or immutable revision does not match configuration."
            )
        model = AutoModel.from_pretrained(resume / "model", local_files_only=True)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = DataLoader(
        _PairDataset(partitions["train"]),
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_seed),
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        _PairDataset(partitions["validation"]),
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    steps_per_epoch = (
        len(train_loader) + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    start_epoch = 0
    global_step = 0
    if resume is not None:
        accelerator.load_state(str(resume / "accelerate-state"))
        progress = json.loads((resume / "progress.json").read_text(encoding="utf-8"))
        start_epoch = int(progress["epoch"])
        global_step = int(progress["global_step"])
        rng_path = resume / "rng.pt"
        if rng_path.exists():
            states = torch.load(rng_path, map_location="cpu", weights_only=False)
            random.setstate(states["python"])
            torch.set_rng_state(states["torch"])
            if torch.cuda.is_available() and states.get("cuda"):
                torch.cuda.set_rng_state_all(states["cuda"])
        if start_epoch >= config.epochs:
            raise ValueError("Resume checkpoint already reached the configured epoch count.")
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install MLflow with the training dependency group.") from exc
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("codesearch-ft-training")
    losses: list[float] = []
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                **config.model_dump(mode="json"),
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "model_id": config.base_model,
                "model_revision": config.model_revision,
            }
        )
        manifest_path = config.output_dir / "dataset-manifest.json"
        write_manifest(manifest, manifest_path)
        if pairs is not None:
            pairs_path = config.output_dir / "training-pairs.json"
            pairs_path.write_text(
                json.dumps(
                    {
                        name: [
                            {
                                "query": row.query,
                                "code": row.code,
                                "query_hash": row.query_hash,
                                "code_hash": row.code_hash,
                                "repository": row.repository,
                                "identity": row.identity,
                                "split": row.split,
                            }
                            for row in values
                        ]
                        for name, values in raw.items()
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        mlflow.log_artifact(str(manifest_path))
        for epoch in range(start_epoch, config.epochs):
            model.train()
            train_loader.generator.manual_seed(config.random_seed + epoch)
            epoch_losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            for batch in train_loader:
                with accelerator.accumulate(model):
                    q = tokenizer(
                        batch["queries"],
                        padding=True,
                        truncation=True,
                        max_length=config.query_max_length,
                        return_tensors="pt",
                    )
                    c = tokenizer(
                        batch["codes"],
                        padding=True,
                        truncation=True,
                        max_length=config.code_max_length,
                        return_tensors="pt",
                    )
                    q = {key: value.to(accelerator.device) for key, value in q.items()}
                    c = {key: value.to(accelerator.device) for key, value in c.items()}
                    qv = _pool(model(**q).last_hidden_state, q["attention_mask"])
                    cv = _pool(model(**c).last_hidden_state, c["attention_mask"])
                    loss = info_nce_loss(
                        qv,
                        cv,
                        config.temperature,
                        query_hashes=batch["query_hashes"],
                        code_hashes=batch["code_hashes"],
                        symmetric=config.symmetric_loss,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    if accelerator.sync_gradients:
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                    epoch_losses.append(float(loss.detach().item()))
            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            val_loss = (
                _evaluate(model, validation_loader, tokenizer, accelerator, config)
                if len(partitions["validation"]) >= 2
                else None
            )
            losses.append(train_loss)
            metrics: dict[str, float] = {"train_loss": train_loss}
            if val_loss is not None:
                metrics["validation_loss"] = val_loss
            mlflow.log_metrics(metrics, step=epoch + 1)
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                accelerator,
                tokenizer,
                config,
                manifest,
                run.info.run_id,
                epoch + 1,
                global_step,
            )
        if partitions["test"]:
            test_loader = DataLoader(
                _PairDataset(partitions["test"]),
                batch_size=config.micro_batch_size,
                shuffle=False,
                collate_fn=_collate,
            )
            test_loader = accelerator.prepare(test_loader)
            test_loss = (
                _evaluate(model, test_loader, tokenizer, accelerator, config)
                if len(partitions["test"]) >= 2
                else None
            )
            if test_loss is not None:
                mlflow.log_metric("test_loss", test_loss)
        mlflow.log_artifact(str(config.output_dir / "export" / "metadata.json"))
        return {
            "run_id": run.info.run_id,
            "train_losses": losses,
            "checkpoint": str(config.output_dir / "export"),
            "manifest_sha256": manifest.manifest_sha256,
            "device": str(accelerator.device),
            "updates": global_step,
        }


class _PairDataset:
    def __init__(self, rows: list[TrainingPair]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainingPair:
        return self.rows[index]


def _read_manifest_pairs(path: Path) -> dict[str, list[TrainingPair]] | None:
    """Load a previously materialized pair cache for offline resumption."""
    data_path = path.with_name("training-pairs.json")
    if not data_path.exists():
        return None
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return {split: [TrainingPair(**row) for row in rows] for split, rows in raw.items()}


def _collate(rows: list[TrainingPair]) -> dict[str, Any]:
    return {
        "queries": [row.query for row in rows],
        "codes": [row.code for row in rows],
        "query_hashes": tuple(row.query_hash for row in rows),
        "code_hashes": tuple(row.code_hash for row in rows),
    }


def _pool(hidden: Any, mask: Any) -> Any:
    import torch.nn.functional as functional

    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return functional.normalize(
        (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1), dim=1
    )


def _evaluate(
    model: Any, loader: Any, tokenizer: Any, accelerator: Any, config: TrainingConfig
) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            q = tokenizer(
                batch["queries"],
                padding=True,
                truncation=True,
                max_length=config.query_max_length,
                return_tensors="pt",
            )
            c = tokenizer(
                batch["codes"],
                padding=True,
                truncation=True,
                max_length=config.code_max_length,
                return_tensors="pt",
            )
            q = {key: value.to(accelerator.device) for key, value in q.items()}
            c = {key: value.to(accelerator.device) for key, value in c.items()}
            qv = _pool(model(**q).last_hidden_state, q["attention_mask"])
            cv = _pool(model(**c).last_hidden_state, c["attention_mask"])
            loss = info_nce_loss(
                qv,
                cv,
                config.temperature,
                query_hashes=batch["query_hashes"],
                code_hashes=batch["code_hashes"],
                symmetric=config.symmetric_loss,
            )
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def _precision(value: str, torch: Any) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    if torch.cuda.is_available():
        return "fp16"
    return "no"


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    accelerator: Any,
    tokenizer: Any,
    config: TrainingConfig,
    manifest: DatasetManifest,
    run_id: str,
    epoch: int,
    global_step: int,
) -> None:
    output = config.output_dir / "checkpoints" / f"epoch-{epoch:03d}"
    output.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(output / "accelerate-state"))
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        model_dir = output / "model"
        unwrapped.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "model_id": config.base_model,
            "model_revision": config.model_revision,
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "dataset_manifest_sha256": manifest.manifest_sha256,
            "preprocessing": "codesearchnet-docstring-free-v1",
            "embedding_pooling": "attention-mask-mean-l2-normalized",
            "embedding_dimension": int(unwrapped.config.hidden_size),
            "run_id": run_id,
            "config": config.model_dump(mode="json"),
        }
        payload = json.dumps(metadata, sort_keys=True, indent=2).encode()
        metadata["metadata_sha256"] = hashlib.sha256(payload).hexdigest()
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / "progress.json").write_text(
            json.dumps({"epoch": epoch, "global_step": global_step}, indent=2) + "\n",
            encoding="utf-8",
        )
        torch = __import__("torch")
        torch.save(
            {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            output / "rng.pt",
        )
        export_dir = config.output_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        exported_model = export_dir / "model"
        if exported_model.exists():
            shutil.rmtree(exported_model)
        unwrapped.save_pretrained(exported_model, safe_serialization=True)
        tokenizer.save_pretrained(exported_model)
        (export_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
