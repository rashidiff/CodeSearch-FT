"""Offline integration tests for the health API and foundation CLI."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from typer.testing import CliRunner

from codesearch_ft.api.app import create_app
from codesearch_ft.cli.app import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_health_api_and_openapi_are_available_without_search_dependencies() -> None:
    with TestClient(create_app()) as client:
        response: Response = client.get("/health")
        openapi_response: Response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "codesearch-ft"
    assert response.json()["search_ready"] is False
    assert "/health" in openapi_response.json()["paths"]


def test_cli_help_describes_foundation_and_phase_two_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "models" in result.output
    assert "config" in result.output
    assert "serve" in result.output
    assert "index" in result.output
    assert "search" in result.output


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["models", "list", "--help"], "model identifiers"),
        (["config", "validate", "--help"], "Configuration schema to validate"),
        (["serve", "--help"], "Bind address"),
    ],
)
def test_cli_command_help_is_available_without_loading_optional_dependencies(
    arguments: list[str], expected: str
) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert expected in result.output


def test_models_list_json_is_catalog_metadata_and_does_not_claim_weights() -> None:
    result = runner.invoke(app, ["models", "list", "--json"])

    assert result.exit_code == 0
    entries = json.loads(result.output)
    assert len(entries) == 1
    assert entries[0]["model_id"] == "microsoft/codebert-base"
    assert entries[0]["weights_checked"] is False


def test_config_validate_cli_validates_without_starting_training() -> None:
    config_path = PROJECT_ROOT / "configs" / "train_cpu_smoke.yaml"
    result = runner.invoke(app, ["config", "validate", str(config_path), "--kind", "training"])

    assert result.exit_code == 0
    assert "profile=cpu-smoke" in result.output
    assert "No work was started" in result.output


def test_config_validate_cli_returns_clear_error_for_invalid_path(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["config", "validate", str(tmp_path / "missing.yaml"), "--kind", "training"],
    )

    assert result.exit_code == 1
    assert "Cannot read configuration" in result.output
