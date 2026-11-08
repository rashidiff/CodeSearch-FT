"""Shared CodeBERT query/code encoder with explicit masked pooling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

from codesearch_ft.domain.errors import ModelUnavailableError
from codesearch_ft.domain.models import Embedding

if TYPE_CHECKING:
    from torch import Tensor

MODEL_ID = "microsoft/codebert-base"
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
PREPROCESSING_VERSION = "code-ast-no-docstring-v1"


def masked_mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool only attended tokens and L2-normalize each row."""
    torch = _torch()
    if last_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError(
            "Expected hidden states [batch, tokens, dimension] and mask [batch, tokens]."
        )
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("Attention mask dimensions must match the hidden-state batch and tokens.")
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Cannot embed an input with no attended tokens.")
    pooled = (last_hidden_state * mask).sum(dim=1) / counts
    norms = torch.linalg.vector_norm(pooled, ord=2, dim=1, keepdim=True)
    if not torch.all(torch.isfinite(pooled)) or torch.any(norms <= 1e-12):
        raise ValueError("Pooled embeddings must be finite and have non-zero norm.")
    return pooled / norms.clamp_min(1e-12)


class CodeBERTEncoder:
    """Encode queries and code through one local-or-explicitly-downloaded model."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
        device: str = "cpu",
        query_max_length: int = 64,
        code_max_length: int = 256,
        batch_size: int = 16,
    ) -> None:
        if min(query_max_length, code_max_length, batch_size) < 1:
            raise ValueError("Maximum lengths and batch size must be positive.")
        torch = _torch()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelUnavailableError(
                "CUDA was requested but is unavailable. Use the default CPU device "
                "or install a supported GPU runtime."
            )
        self.tokenizer = tokenizer
        self.model = model
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = torch.device(device)
        self.query_max_length = query_max_length
        self.code_max_length = code_max_length
        self.batch_size = batch_size
        self._lock = RLock()
        self.dimension = int(model.config.hidden_size)
        if self.dimension < 1:
            raise ValueError("Model hidden size must be positive.")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
        query_max_length: int = 64,
        code_max_length: int = 256,
        batch_size: int = 16,
    ) -> CodeBERTEncoder:
        """Load the pinned model; network access requires explicit opt-in."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailableError(
                "CodeBERT inference dependencies are missing. "
                "Install with `uv sync --extra inference`."
            ) from exc
        checkpoint_path = Path(model_id)
        metadata_path = checkpoint_path / "metadata.json"
        weights_path = checkpoint_path
        checkpoint_metadata: dict[str, Any] = {}
        if checkpoint_path.is_dir():
            if not metadata_path.exists() and (checkpoint_path.parent / "metadata.json").exists():
                metadata_path = checkpoint_path.parent / "metadata.json"
            if metadata_path.exists():
                checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (checkpoint_path / "model").is_dir():
                weights_path = checkpoint_path / "model"
        if checkpoint_metadata:
            expected_metadata_sha256 = checkpoint_metadata.get("metadata_sha256")
            if expected_metadata_sha256 is not None:
                unsigned_metadata = dict(checkpoint_metadata)
                unsigned_metadata.pop("metadata_sha256", None)
                payload = json.dumps(unsigned_metadata, sort_keys=True, indent=2).encode()
                actual_metadata_sha256 = hashlib.sha256(payload).hexdigest()
                if actual_metadata_sha256 != expected_metadata_sha256:
                    raise ModelUnavailableError(
                        "Fine-tuned checkpoint metadata failed SHA-256 validation."
                    )
            if checkpoint_metadata.get("model_id") != MODEL_ID:
                raise ModelUnavailableError(
                    "Fine-tuned checkpoint base model metadata is incompatible."
                )
            revision = str(checkpoint_metadata.get("model_revision", ""))
            if len(revision) != 40:
                raise ModelUnavailableError("Fine-tuned checkpoint revision metadata is invalid.")
        try:
            tokenizer = cast(Any, AutoTokenizer).from_pretrained(
                str(weights_path),
                revision=None if checkpoint_path.is_dir() else revision,
                local_files_only=local_files_only,
            )
            model = cast(Any, AutoModel).from_pretrained(
                str(weights_path),
                revision=None if checkpoint_path.is_dir() else revision,
                local_files_only=local_files_only,
            )
        except OSError as exc:
            if local_files_only:
                raise ModelUnavailableError(
                    f"CodeBERT weights for revision {revision} are not cached locally. "
                    "Run `codesearch models download` explicitly or set up the Hugging Face cache."
                ) from exc
            raise ModelUnavailableError(
                f"Unable to load CodeBERT model {model_id}@{revision}: {exc}"
            ) from exc
        if checkpoint_metadata:
            if checkpoint_metadata.get("embedding_pooling") != "attention-mask-mean-l2-normalized":
                raise ModelUnavailableError(
                    "Fine-tuned checkpoint pooling metadata is incompatible."
                )
            if checkpoint_metadata.get("embedding_dimension") != int(model.config.hidden_size):
                raise ModelUnavailableError(
                    "Fine-tuned checkpoint embedding dimension is incompatible."
                )
        return cls(
            tokenizer,
            model,
            model_id=model_id,
            model_revision=revision,
            device=device,
            query_max_length=query_max_length,
            code_max_length=code_max_length,
            batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return self._encode(texts, self.query_max_length)

    def encode_code(self, snippets: Sequence[str]) -> tuple[Embedding, ...]:
        return self._encode(snippets, self.code_max_length)

    def _encode(self, texts: Sequence[str], max_length: int) -> tuple[Embedding, ...]:
        output: list[Embedding] = []
        torch = _torch()
        with self._lock:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                tokenized = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokenized = {name: tensor.to(self.device) for name, tensor in tokenized.items()}
                with torch.inference_mode():
                    model_output = self.model(**tokenized)
                    embeddings = masked_mean_pool(
                        model_output.last_hidden_state, tokenized["attention_mask"]
                    )
                if embeddings.shape[1] != self.dimension:
                    raise ValueError(
                        "Model output dimension differs from its configured hidden size."
                    )
                rows = cast(list[list[float]], cast(Any, embeddings.cpu()).tolist())
                output.extend(tuple(row) for row in rows)
        return tuple(output)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ModelUnavailableError(
            "PyTorch is missing. Install CodeSearch-FT inference dependencies "
            "with `uv sync --extra inference`."
        ) from exc
    return torch
