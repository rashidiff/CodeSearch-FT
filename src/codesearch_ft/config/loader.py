"""Safe YAML loading and validation for training and evaluation settings."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from codesearch_ft.config.models import EvaluationConfig, TrainingConfig
from codesearch_ft.domain.errors import ConfigurationError

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config(path: Path, config_type: type[ConfigT]) -> ConfigT:
    """Load a YAML mapping and validate it without echoing input values."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read configuration file '{path}': {exc.strerror}."
        ) from exc

    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None) or "invalid YAML syntax"
        raise ConfigurationError(f"Invalid YAML in '{path}': {problem}.") from exc

    if not isinstance(raw_data, dict):
        raise ConfigurationError(f"Configuration '{path}' must contain a YAML mapping.")

    try:
        return config_type.model_validate(raw_data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise ConfigurationError(f"Invalid configuration '{path}': {details}.") from exc


def load_training_config(path: Path) -> TrainingConfig:
    """Load a CPU-smoke or GPU training profile."""
    return load_config(path, TrainingConfig)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load an evaluation profile."""
    return load_config(path, EvaluationConfig)
