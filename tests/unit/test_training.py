from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from codesearch_ft.config.loader import load_training_config
from codesearch_ft.datasets.codesearchnet import (
    TrainingPair,
    iter_source_splits,
    near_duplicate_count,
    prepare_partitions,
    remove_top_level_docstring,
    sample_pairs,
    validate_row,
    write_manifest,
)
from codesearch_ft.ml.contrastive import info_nce_loss


def _pair(query: str, code: str, repository: str, split: str) -> TrainingPair:
    import hashlib

    qh = hashlib.sha256(query.casefold().encode()).hexdigest()
    ch = hashlib.sha256(code.casefold().encode()).hexdigest()
    return TrainingPair(query, code, qh, ch, repository, f"{repository}/x.py::f", split)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repository_name": "org/repo",
        "func_path_in_repository": "file.py",
        "func_name": "f",
        "whole_func_string": 'def f():\n    """Useful function does something."""\n    return 1\n',
        "language": "python",
        "func_code_string": 'def f():\n    """Useful function does something."""\n    return 1\n',
        "func_documentation_string": "Useful function does something.",
        "split_name": "train",
        "func_code_url": "https://example.test/file.py",
    }
    row.update(updates)
    return row


def test_dataset_row_schema_and_docstring_removal() -> None:
    pair = validate_row(_row(), "train")
    assert pair is not None
    assert '"""' not in pair.code
    assert pair.code.endswith("return 1")
    with pytest.raises(ValueError, match="missing required columns"):
        validate_row({"language": "python"}, "train")


def test_docstring_removal_variants() -> None:
    assert "doc" not in remove_top_level_docstring('def f():\n    r"""doc"""\n    return 1\n')
    code = "def f():\n    return 1\n"
    assert remove_top_level_docstring(code) == code


def test_cross_split_exact_duplicates_are_quarantined(tmp_path: Path) -> None:
    train = _pair("same query here", "def f():\n  return 1", "org/a", "train")
    validation = _pair("same query here", "def g():\n  return 2", "org/b", "validation")
    test = _pair("unique query here", "def h():\n  return 3", "org/c", "test")
    config = load_training_config(Path("configs/train_cpu_smoke.yaml"))
    cleaned, manifest = prepare_partitions(
        {"train": [train], "validation": [validation], "test": [test]}, config
    )
    assert cleaned["train"] == []
    assert cleaned["validation"] == []
    assert cleaned["test"] == [test]
    assert manifest.cross_split_duplicate_groups == 1
    assert len(manifest.manifest_sha256) == 64
    assert manifest.source_url == "https://huggingface.co/datasets/code-search-net/code_search_net"
    assert "license=other" in manifest.license_caveat
    assert len(manifest.dataset_sha256) == 64
    cleaned_again, manifest_again = prepare_partitions(
        {"train": [train], "validation": [validation], "test": [test]}, config
    )
    assert cleaned_again == cleaned
    assert manifest_again == manifest
    path = tmp_path / "dataset-manifest.json"
    write_manifest(manifest, path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["dataset_sha256"] == manifest.dataset_sha256


def test_near_duplicate_count_uses_stable_candidate_generation() -> None:
    rows = [
        _pair("query words here", "def f():\n return alpha beta gamma", "a", "train"),
        _pair("other query words", "def g():\n return alpha beta gamma", "b", "train"),
        _pair("different query", "def h():\n return completely different", "c", "train"),
    ]
    assert near_duplicate_count(rows, 0.6) >= 1


def test_infonce_shapes_gradients_duplicates_and_single_pair() -> None:
    torch = pytest.importorskip("torch")
    query = torch.randn(3, 8, requires_grad=True)
    code = torch.randn(3, 8, requires_grad=True)
    loss = info_nce_loss(
        query, code, 0.05, query_hashes=("a", "a", "b"), code_hashes=("x", "y", "z"), symmetric=True
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert query.grad is not None and code.grad is not None
    one = info_nce_loss(torch.randn(1, 8), torch.randn(1, 8), 0.05)
    assert torch.isfinite(one)


def test_training_profiles_include_immutable_dataset_source() -> None:
    for name in ("train_cpu_smoke.yaml", "train_gpu_pilot.yaml", "train_gpu_full.yaml"):
        config = load_training_config(Path("configs") / name)
        assert len(config.model_revision) == 40
        assert len(config.dataset_revision) == 40
        assert config.dataset_config == "python"


def test_deterministic_sampling_and_unsupported_dataset_revision() -> None:
    rows = [
        _pair(f"query example {index}", f"def f{index}():\n return {index}", "r", "train")
        for index in range(12)
    ]
    assert sample_pairs(rows, 5, 77) == sample_pairs(rows, 5, 77)
    config = load_training_config(Path("configs/train_cpu_smoke.yaml")).model_copy(
        update={"dataset_revision": "0" * 40}
    )
    with pytest.raises(ValueError, match="Unsupported CodeSearchNet"):
        iter_source_splits(config, {"train": 1})


def test_stream_adapter_validates_schema_and_respects_scan_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = pytest.importorskip("datasets")

    class FakeStream:
        features = {name: str for name in _row()}

        def __iter__(self):
            yield _row()
            yield _row(func_documentation_string="too short")
            yield _row(func_name="second")

    calls: list[dict[str, object]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> FakeStream:
        calls.append(kwargs)
        data_files = cast(dict[str, object], kwargs["data_files"])
        split = next(iter(data_files))
        split_name = "valid" if split == "validation" else split

        class SplitStream(FakeStream):
            def __iter__(self):
                yield _row(split_name=split_name)
                yield _row(split_name=split_name, func_documentation_string="too short")
                yield _row(split_name=split_name, func_name="second")

        return SplitStream()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    config = load_training_config(Path("configs/train_cpu_smoke.yaml")).model_copy(
        update={"dataset_cache_dir": tmp_path}
    )
    result = iter_source_splits(config, {"train": 1, "validation": 1, "test": 1})
    assert all(len(result[split]) == 1 for split in result)
    assert len(calls) == 3
    assert all(call["streaming"] is True for call in calls)
    assert all(
        "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555" in str(call["data_files"]) for call in calls
    )

    class BadStream:
        features = {"language": str}

        def __iter__(self):
            return iter(())

    def bad_load_dataset(*args: object, **kwargs: object) -> BadStream:
        return BadStream()

    monkeypatch.setattr(datasets, "load_dataset", bad_load_dataset)
    with pytest.raises(ValueError, match="Unsupported CodeSearchNet schema"):
        iter_source_splits(config, {"train": 1})


def test_training_checkpoint_and_resume_with_offline_model_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from codesearch_ft.ml.trainer import train

    class FixtureTokenizer:
        def __call__(
            self,
            texts: list[str],
            *,
            max_length: int,
            **_: Any,
        ) -> dict[str, Any]:
            tokens = [
                [
                    int.from_bytes(hashlib.sha256(token.encode()).digest()[:2]) % 127 + 1
                    for token in text.split()[:max_length]
                ]
                for text in texts
            ]
            width = max(map(len, tokens), default=1)
            input_ids = [row + [0] * (width - len(row)) for row in tokens]
            mask = [[int(token != 0) for token in row] for row in input_ids]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }

        def save_pretrained(self, path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer-fixture.json").write_text("{}", encoding="utf-8")

    class FixtureModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.embedding = torch.nn.Embedding(128, 8)
            self.config = SimpleNamespace(hidden_size=8)

        def forward(self, input_ids: Any, **_: Any) -> Any:
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        def save_pretrained(self, path: Path, *, safe_serialization: bool = True) -> None:
            del safe_serialization
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "fixture-state.pt")

    def fixture_model(
        model_id: str | Path, *, revision: str | None = None, **_: Any
    ) -> FixtureModel:
        del revision
        model = FixtureModel()
        saved_state = Path(model_id) / "fixture-state.pt"
        if saved_state.exists():
            model.load_state_dict(torch.load(saved_state, map_location="cpu", weights_only=True))
        return model

    def fixture_tokenizer(*_: Any, **__: Any) -> FixtureTokenizer:
        return FixtureTokenizer()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fixture_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fixture_tokenizer)
    output = tmp_path / "checkpoints"
    config = load_training_config(Path("configs/train_cpu_smoke.yaml")).model_copy(
        update={
            "epochs": 1,
            "micro_batch_size": 2,
            "max_train_samples": 4,
            "validation_max_samples": 2,
            "test_max_samples": 2,
            "output_dir": output,
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        }
    )
    pairs: dict[str, list[TrainingPair]] = {}
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        rows: list[TrainingPair] = []
        for index in range(count):
            query = f"{split} operation query number {index}"
            code = f"def operation_{split}_{index}(value):\n    return value + {index}"
            rows.append(
                TrainingPair(
                    query,
                    code,
                    hashlib.sha256(query.casefold().encode()).hexdigest(),
                    hashlib.sha256(code.casefold().encode()).hexdigest(),
                    f"repository-{split}-{index}",
                    f"repository-{split}-{index}/module.py::operation_{index}",
                    split,
                )
            )
        pairs[split] = rows
    result = train(config, pairs=pairs)
    assert result["updates"] == 2
    assert all(value >= 0 for value in result["train_losses"])
    checkpoint = output / "checkpoints" / "epoch-001"
    assert (checkpoint / "accelerate-state" / "optimizer.bin").exists()
    assert (output / "export" / "metadata.json").exists()
    resumed = train(config.model_copy(update={"epochs": 2}), resume=checkpoint)
    assert resumed["updates"] == 4
    assert (output / "checkpoints" / "epoch-002" / "progress.json").exists()
