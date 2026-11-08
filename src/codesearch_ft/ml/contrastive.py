"""InfoNCE objectives over normalized paired query and code embeddings."""

from __future__ import annotations

from typing import Any


def info_nce_loss(
    query_embeddings: Any,
    code_embeddings: Any,
    temperature: float,
    *,
    query_hashes: tuple[str, ...] | None = None,
    code_hashes: tuple[str, ...] | None = None,
    symmetric: bool = False,
) -> Any:
    """Compute duplicate-aware in-batch cross entropy with diagonal positives."""
    import torch
    import torch.nn.functional as functional

    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive.")
    if query_embeddings.ndim != 2 or code_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-two [batch, dimension] tensors.")
    if query_embeddings.shape != code_embeddings.shape or query_embeddings.shape[0] < 1:
        raise ValueError("Query and code embedding batches must have equal nonzero shapes.")
    batch_size = query_embeddings.shape[0]
    if query_hashes is not None and len(query_hashes) != batch_size:
        raise ValueError("Query hash count must match batch size.")
    if code_hashes is not None and len(code_hashes) != batch_size:
        raise ValueError("Code hash count must match batch size.")
    queries = functional.normalize(query_embeddings.float(), p=2, dim=1)
    codes = functional.normalize(code_embeddings.float(), p=2, dim=1)
    logits = queries @ codes.T / temperature
    targets = torch.arange(batch_size, device=logits.device)
    duplicate_mask = torch.zeros_like(logits, dtype=torch.bool)
    for row in range(batch_size):
        for column in range(batch_size):
            if row != column and (
                (query_hashes is not None and query_hashes[row] == query_hashes[column])
                or (code_hashes is not None and code_hashes[row] == code_hashes[column])
            ):
                duplicate_mask[row, column] = True
    logits = logits.masked_fill(duplicate_mask, torch.finfo(logits.dtype).min)
    forward = functional.cross_entropy(logits, targets)
    if not symmetric:
        return forward
    reverse_logits = (codes @ queries.T / temperature).masked_fill(
        duplicate_mask.T, torch.finfo(logits.dtype).min
    )
    return (forward + functional.cross_entropy(reverse_logits, targets)) / 2
