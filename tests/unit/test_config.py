"""Offline validation tests for shipped configuration profiles."""

from pathlib import Path

import pytest

from codesearch_ft.config.loader import load_evaluation_config, load_training_config
from codesearch_ft.domain.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("filename", "expected_profile"),
    [
        ("train_cpu_smoke.yaml", "cpu-smoke"),
        ("train_gpu_pilot.yaml", "gpu-pilot"),
        ("train_gpu_full.yaml", "gpu-full"),
    ],
)
def test_training_profiles_validate(filename: str, expected_profile: str) -> None:
    config = load_training_config(PROJECT_ROOT / "configs" / filename)

    assert config.profile == expected_profile
    assert config.base_model == "microsoft/codebert-base"


def test_evaluation_profile_validates_without_requiring_data_files() -> None:
    config = load_evaluation_config(PROJECT_ROOT / "configs" / "evaluation.yaml")

    assert config.profile == "evaluation"
    assert config.top_k == 10
    assert config.candidate_corpus == Path(
        "artifacts/evaluation/codesearchnet-test-v1/candidates.jsonl"
    )


def test_invalid_training_values_raise_actionable_error_without_input_values(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        "profile: cpu-smoke\nlearning_rate: -4\nprivate_token: do-not-print\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as raised:
        load_training_config(config_file)

    message = str(raised.value)
    assert "learning_rate" in message
    assert "private_token" in message
    assert "do-not-print" not in message


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    config_file = tmp_path / "broken.yaml"
    config_file.write_text("profile: [unfinished", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_training_config(config_file)


def test_non_mapping_configuration_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- not-a-mapping", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="YAML mapping"):
        load_training_config(config_file)


def test_missing_configuration_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        load_training_config(tmp_path / "missing.yaml")
