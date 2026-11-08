"""Bounded CodeSearchNet Python streaming and leakage-aware preprocessing."""

# datasets' dynamic IterableDataset schema is runtime validated below.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codesearch_ft.config.models import TrainingConfig

DATASET_ID = "code-search-net/code_search_net"
DATASET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
REQUIRED_COLUMNS = frozenset(
    {
        "repository_name",
        "func_path_in_repository",
        "func_name",
        "whole_func_string",
        "language",
        "func_code_string",
        "func_documentation_string",
        "split_name",
        "func_code_url",
    }
)
_DOCSTRING = re.compile(r"(?m)^(\s*)([rRuUbBfF]{0,2})(\"\"\"|''')[\s\S]*?\3\s*(?:\r?\n)?")


@dataclass(frozen=True, slots=True)
class TrainingPair:
    query: str
    code: str
    query_hash: str
    code_hash: str
    repository: str
    identity: str
    split: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    source_url: str
    revision: str
    config: str
    license_caveat: str
    filters: tuple[str, ...]
    seed: int
    row_limits: dict[str, int | None]
    max_source_scan_rows: dict[str, int | None]
    row_counts: dict[str, int]
    kept_counts: dict[str, int]
    removed_counts: dict[str, int]
    exact_duplicate_groups: int
    near_duplicate_groups: int
    near_duplicate_algorithm: str
    cross_split_duplicate_groups: int
    repository_overlap_count: int
    policy: str
    preprocessing: str
    config_sha256: str
    dataset_sha256: str
    manifest_sha256: str


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for stable duplicate identities."""
    return " ".join(text.casefold().split())


def _digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def remove_top_level_docstring(code: str) -> str:
    """Remove the function's leading docstring while preserving its implementation."""
    lines = code.splitlines(keepends=True)
    index = 0
    if lines and lines[0].lstrip().startswith(("def ", "async def ")):
        index = 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return code
    match = re.match(r"^(\s*)([rRuUbBfF]{0,2})(\"\"\"|''')", lines[index])
    if match is None:
        return code
    delimiter = match.group(3)
    end = index
    while end < len(lines):
        if delimiter in lines[end][match.end() if end == index else 0 :]:
            break
        end += 1
    if end == len(lines):
        return code
    return "".join(lines[:index] + lines[end + 1 :])


def validate_row(row: Mapping[str, Any], split: str) -> TrainingPair | None:
    """Convert a validated source row into a docstring-free query/code pair."""
    if not REQUIRED_COLUMNS.issubset(row):
        missing = sorted(REQUIRED_COLUMNS.difference(row))
        raise ValueError(f"CodeSearchNet row is missing required columns: {missing}.")
    if row["language"] != "python" or row["split_name"] not in {
        split,
        "valid" if split == "validation" else split,
    }:
        return None
    query = str(row["func_documentation_string"] or "").strip()
    code = str(row["func_code_string"] or "").strip()
    code = remove_top_level_docstring(code).strip()
    if len(query.split()) < 3 or len(code.splitlines()) < 2:
        return None
    repository = str(row["repository_name"] or "").strip().casefold()
    path = str(row["func_path_in_repository"] or "").strip().casefold()
    function = str(row["func_name"] or "").strip().casefold()
    identity = f"{repository}/{path}::{function}"
    return TrainingPair(query, code, _digest(query), _digest(code), repository, identity, split)


def iter_source_splits(
    config: TrainingConfig, split_limits: Mapping[str, int | None]
) -> dict[str, list[TrainingPair]]:
    """Load bounded source examples from immutable Hugging Face Parquet files."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the training dependencies with `uv sync --extra inference --group training`."
        ) from exc
    if config.dataset_id != DATASET_ID or config.dataset_revision != DATASET_REVISION:
        raise ValueError("Unsupported CodeSearchNet dataset ID or immutable revision.")
    output: dict[str, list[TrainingPair]] = {}
    for split in ("train", "validation", "test"):
        limit = split_limits.get(split)
        scan_limit = None if limit is None else max(limit * 20, limit)
        filename = f"{config.dataset_config}/{split}-00000-of-00001.parquet"
        stream = load_dataset(
            "parquet",
            data_files={
                split: f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{filename}"
            },
            split=split,
            streaming=True,
            cache_dir=str(config.dataset_cache_dir),
        )
        columns = set(stream.features)
        if not REQUIRED_COLUMNS.issubset(columns):
            missing_columns = sorted(REQUIRED_COLUMNS - columns)
            raise ValueError(
                f"Unsupported CodeSearchNet schema; missing columns: {missing_columns}."
            )
        rows: list[TrainingPair] = []
        for row_index, row in enumerate(stream):
            pair = validate_row(row, split)
            if pair is not None:
                rows.append(pair)
            if limit is not None and len(rows) >= limit:
                break
            if scan_limit is not None and row_index + 1 >= scan_limit:
                break
        output[split] = rows
    return output


def sample_pairs(pairs: Iterable[TrainingPair], limit: int | None, seed: int) -> list[TrainingPair]:
    """Deterministically sample a bounded dataset without changing its split."""
    items = list(pairs)
    if limit is None or len(items) <= limit:
        return items
    return random.Random(seed).sample(items, limit)


def prepare_partitions(
    raw: Mapping[str, list[TrainingPair]], config: TrainingConfig
) -> tuple[dict[str, list[TrainingPair]], DatasetManifest]:
    """Remove cross-split duplicates and report near duplicates without moving rows."""
    if set(raw) != {"train", "validation", "test"}:
        raise ValueError("Dataset must preserve train, validation, and test partitions.")
    sampled = {
        "train": sample_pairs(raw["train"], config.max_train_samples, config.random_seed),
        "validation": sample_pairs(
            raw["validation"], config.validation_max_samples, config.random_seed + 1
        ),
        "test": sample_pairs(raw["test"], config.test_max_samples, config.random_seed + 2),
    }
    hash_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split, pairs in sampled.items():
        for pair in pairs:
            hash_splits[("query", pair.query_hash)].add(split)
            hash_splits[("code", pair.code_hash)].add(split)
    leaked = {key for key, splits in hash_splits.items() if len(splits) > 1}
    removed: dict[str, list[TrainingPair]] = {name: [] for name in sampled}
    cleaned: dict[str, list[TrainingPair]] = {}
    for split, pairs in sampled.items():
        for pair in pairs:
            if ("query", pair.query_hash) in leaked or ("code", pair.code_hash) in leaked:
                removed[split].append(pair)
        cleaned[split] = [pair for pair in pairs if pair not in removed[split]]
    all_pairs = [pair for values in sampled.values() for pair in values]
    exact_groups = len(all_pairs) - len({pair.code_hash for pair in all_pairs})
    near_groups = near_duplicate_count(all_pairs, config.near_duplicate_threshold)
    repositories = {
        split: {pair.repository for pair in values} for split, values in cleaned.items()
    }
    overlaps = (
        (repositories["train"] & repositories["validation"])
        | (repositories["train"] & repositories["test"])
        | (repositories["validation"] & repositories["test"])
    )
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    data_identity = [
        (split, pair.query_hash, pair.code_hash, pair.identity)
        for split in ("train", "validation", "test")
        for pair in cleaned[split]
    ]
    dataset_hash = hashlib.sha256(
        json.dumps(data_identity, separators=(",", ":")).encode()
    ).hexdigest()
    source = {
        "dataset_id": config.dataset_id,
        "source_url": f"https://huggingface.co/datasets/{config.dataset_id}",
        "revision": config.dataset_revision,
        "config": config.dataset_config,
        "license_caveat": (
            "Dataset card declares license=other; examples originate in GitHub repositories "
            "and do not include per-example license metadata."
        ),
        "filters": (
            "language == python",
            "source split matches train/valid/test assignment",
            "query has at least three whitespace-delimited tokens",
            "docstring-free code has at least two lines",
        ),
        "seed": config.random_seed,
        "row_limits": {
            "train": config.max_train_samples,
            "validation": config.validation_max_samples,
            "test": config.test_max_samples,
        },
        "max_source_scan_rows": {
            name: None if limit is None else max(limit * 20, limit)
            for name, limit in {
                "train": config.max_train_samples,
                "validation": config.validation_max_samples,
                "test": config.test_max_samples,
            }.items()
        },
        "row_counts": {name: len(raw[name]) for name in raw},
        "kept_counts": {name: len(values) for name, values in cleaned.items()},
        "removed_counts": {name: len(values) for name, values in removed.items()},
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_groups": near_groups,
        "near_duplicate_algorithm": (
            f"blake2b-minhash-64-bands-16-width-4-max-bucket-256-jaccard-threshold-"
            f"{config.near_duplicate_threshold:g}-v1"
        ),
        "cross_split_duplicate_groups": len(leaked),
        "repository_overlap_count": len(overlaps),
        "policy": (
            "Quarantine all exact query/code duplicate groups spanning official splits; "
            "preserve remaining assignments."
        ),
        "preprocessing": (
            "Normalize duplicate identities, strip leading triple-quoted docstrings, "
            "report token-set Jaccard near duplicates using configured "
            "64-permutation MinHash bands, candidate bucket cap 256, "
            "and exact Jaccard confirmation; "
            "report repository overlap."
        ),
        "config_sha256": config_hash,
        "dataset_sha256": dataset_hash,
    }
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    manifest = DatasetManifest(**source, manifest_sha256=hashlib.sha256(encoded).hexdigest())
    return cleaned, manifest


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Persist auditable dataset preparation metadata, never raw examples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def near_duplicate_count(pairs: list[TrainingPair], threshold: float) -> int:
    """Count near duplicates with deterministic MinHash bands and exact Jaccard checks."""
    tokens = [set(normalize_text(pair.code).split()) for pair in pairs]
    bands: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    permutations = 64
    band_width = 4
    prime = 2**61 - 1
    seeds = tuple(range(1, permutations + 1))
    for index, row_tokens in enumerate(tokens):
        if not row_tokens:
            continue
        signature = tuple(
            min(
                int.from_bytes(
                    hashlib.blake2b(f"{seed}:{token}".encode(), digest_size=8).digest(),
                    "big",
                )
                % prime
                for token in row_tokens
            )
            for seed in seeds
        )
        for band in range(permutations // band_width):
            start = band * band_width
            bands[(band, signature[start : start + band_width])].append(index)
    candidates = {
        pair
        for members in bands.values()
        if len(members) <= 256
        for pair in itertools.combinations(members, 2)
    }
    return sum(
        bool(union) and len(tokens[left] & tokens[right]) / len(union) >= threshold
        for left, right in candidates
        for union in (tokens[left] | tokens[right],)
    )
