"""Offline PyTorch tests for masked pooling and shared CodeBERT encoding."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import Tensor

from codesearch_ft.domain.errors import ModelUnavailableError
from codesearch_ft.ml.codebert_encoder import CodeBERTEncoder, masked_mean_pool


def test_masked_mean_pool_excludes_padding_and_l2_normalizes() -> None:
    hidden = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 4.0], [1000.0, 1000.0]],
            [[3.0, 0.0], [0.0, 4.0], [-500.0, 600.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    embeddings = masked_mean_pool(hidden, mask)

    assert embeddings.shape == (2, 2)
    assert torch.allclose(embeddings[0], embeddings[1])
    rows = embeddings.numpy().tolist()
    norms = [sum(float(value) ** 2 for value in row) ** 0.5 for row in rows]
    assert all(abs(norm - 1.0) < 1e-6 for norm in norms)


def test_masked_pool_validates_shapes_and_empty_masks() -> None:
    with pytest.raises(ValueError, match="Expected hidden states"):
        masked_mean_pool(torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="dimensions must match"):
        masked_mean_pool(torch.ones(1, 2, 3), torch.ones(1, 1))
    with pytest.raises(ValueError, match="no attended tokens"):
        masked_mean_pool(torch.ones(1, 2, 3), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="non-zero norm"):
        masked_mean_pool(torch.zeros(1, 2, 3), torch.ones(1, 2))


class TinyTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, Tensor]:
        assert padding and truncation and return_tensors == "pt"
        lengths = [min(len(text.split()) + 1, max_length) for text in texts]
        width = max(lengths)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = row + 1
            ids[row, length:] = 999
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class TinyModel:
    config = SimpleNamespace(hidden_size=3)

    def to(self, device: torch.device) -> TinyModel:
        assert device.type == "cpu"
        return self

    def eval(self) -> TinyModel:
        return self

    def __call__(self, input_ids: Tensor, attention_mask: Tensor) -> SimpleNamespace:
        del attention_mask
        values = input_ids.to(torch.float32).unsqueeze(-1)
        hidden = torch.cat((values, values * 2, values * 3), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_encoder_batches_both_inputs_and_returns_normalized_dimensioned_vectors() -> None:
    encoder = CodeBERTEncoder(
        TinyTokenizer(), TinyModel(), query_max_length=4, code_max_length=8, batch_size=2
    )

    queries = encoder.encode_queries(("short", "two words here", "last query"))
    snippets = encoder.encode_code(("def one(): pass", "def two(): pass"))

    assert len(queries) == 3
    assert len(snippets) == 2
    assert all(len(embedding) == encoder.dimension == 3 for embedding in (*queries, *snippets))
    assert all(abs(sum(value * value for value in embedding) - 1.0) < 1e-6 for embedding in queries)
    assert queries == encoder.encode_queries(("short", "two words here", "last query"))


def test_encoder_rejects_invalid_batch_settings() -> None:
    with pytest.raises(ValueError, match="positive"):
        CodeBERTEncoder(TinyTokenizer(), TinyModel(), batch_size=0)


def test_missing_local_checkpoint_has_actionable_error_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingWeights:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> None:
            raise OSError("not found in local cache")

    transformers = ModuleType("transformers")
    transformers.AutoModel = MissingWeights  # type: ignore[attr-defined]
    transformers.AutoTokenizer = MissingWeights  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(ModelUnavailableError, match="not cached locally"):
        CodeBERTEncoder.from_pretrained(local_files_only=True)
