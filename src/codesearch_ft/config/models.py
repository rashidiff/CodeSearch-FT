"""Pydantic v2 models for the documented experiment profiles."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfig(BaseModel):
    """Validated settings consumed by the later training implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["cpu-smoke", "gpu-pilot", "gpu-full"]
    base_model: str = "microsoft/codebert-base"
    model_revision: str = "3b0952feddeffad0063f274080e3c23d75e7eb39"
    dataset_id: str = "code-search-net/code_search_net"
    dataset_revision: str = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
    dataset_config: str = "python"
    dataset_cache_dir: Path = Path("data/cache/codesearchnet")
    validation_max_samples: int | None = Field(default=256, ge=1)
    test_max_samples: int | None = Field(default=None, ge=1)
    near_duplicate_threshold: float = Field(default=0.92, gt=0, le=1)
    symmetric_loss: bool = False
    query_max_length: int = Field(default=64, ge=1, le=512)
    code_max_length: int = Field(default=256, ge=1, le=2048)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    temperature: float = Field(default=0.05, gt=0)
    micro_batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    epochs: int = Field(default=2, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    random_seed: int = Field(default=42, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    mixed_precision: Literal["no", "fp16", "bf16", "auto"] = "no"
    max_train_samples: int | None = Field(default=None, ge=1)
    output_dir: Path = Path("artifacts/checkpoints")
    tracking_uri: str = "sqlite:///artifacts/mlflow.db"

    @model_validator(mode="after")
    def validate_approved_sources(self) -> "TrainingConfig":
        if self.base_model != "microsoft/codebert-base":
            raise ValueError("Phase 4 training supports only microsoft/codebert-base.")
        if len(self.model_revision) != 40 or len(self.dataset_revision) != 40:
            raise ValueError("Model and dataset revisions must be immutable 40-character commits.")
        if self.dataset_config != "python":
            raise ValueError("Only the CodeSearchNet Python dataset config is supported.")
        return self


class EvaluationConfig(BaseModel):
    """Validated evaluation inputs and metric cutoffs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["evaluation"]
    candidate_corpus: Path
    queries: Path
    output_dir: Path = Path("artifacts/evaluation")
    dataset_id: str = "code-search-net/code_search_net"
    dataset_revision: str = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
    dataset_config: str = "python"
    model_revision: str = "3b0952feddeffad0063f274080e3c23d75e7eb39"
    fine_tuned_checkpoint: Path = Path("artifacts/evaluation/fine-tuned-codebert")
    candidate_limit: int = Field(default=128, ge=2, le=10_000)
    query_limit: int = Field(default=128, ge=1, le=10_000)
    top_k: int = Field(default=10, ge=10, le=10)
    random_seed: int = Field(default=42, ge=0)
    rrf_constant: int = Field(default=60, ge=1)
    modes: list[Literal["bm25", "pretrained_codebert", "fine_tuned_codebert", "hybrid"]] = Field(
        default_factory=lambda: [
            "bm25",
            "pretrained_codebert",
            "fine_tuned_codebert",
            "hybrid",
        ]
    )

    @model_validator(mode="after")
    def validate_fixed_pool(self) -> "EvaluationConfig":
        if self.query_limit > self.candidate_limit:
            raise ValueError("query_limit must not exceed candidate_limit for paired evaluation.")
        if self.dataset_config != "python":
            raise ValueError("Only the CodeSearchNet Python evaluation config is supported.")
        if len(self.dataset_revision) != 40 or len(self.model_revision) != 40:
            raise ValueError("Dataset and model revisions must be immutable 40-character commits.")
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("Evaluation modes must be non-empty and unique.")
        if "hybrid" in self.modes and not {
            "bm25",
            "fine_tuned_codebert",
        }.issubset(self.modes):
            raise ValueError("Hybrid evaluation requires BM25 and fine-tuned CodeBERT modes.")
        return self
