"""The fixed benchmark remains an explicit CLI action."""

from pathlib import Path

from typer.testing import CliRunner

from codesearch_ft.cli.app import app


def test_benchmark_help_describes_explicit_configured_execution() -> None:
    result = CliRunner().invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    assert "Fixed-pool evaluation YAML" in result.output


def test_evaluation_config_validates_without_running_models_or_data() -> None:
    config = Path(__file__).resolve().parents[2] / "configs" / "evaluation.yaml"
    result = CliRunner().invoke(
        app,
        ["config", "validate", str(config), "--kind", "evaluation"],
    )

    assert result.exit_code == 0
    assert "kind=evaluation" in result.output
    assert "No work was started" in result.output
