"""Offline API surface and resource-bound contract checks."""

from typer.testing import CliRunner

from codesearch_ft.api.app import create_app
from codesearch_ft.api.schemas import SearchRequest
from codesearch_ft.cli.app import app


def test_api_exposes_only_health_catalog_and_bounded_search_routes() -> None:
    paths = create_app().openapi()["paths"]

    assert set(paths) == {"/health", "/v1/models", "/v1/indexes", "/v1/search"}
    assert paths["/v1/search"].keys() == {"post"}


def test_search_contract_rejects_extra_fields_and_bounds_inputs() -> None:
    query_constraints = SearchRequest.model_fields["query"].metadata
    top_k_constraints = SearchRequest.model_fields["top_k"].metadata

    assert SearchRequest.model_config.get("extra") == "forbid"
    assert any(getattr(item, "min_length", None) == 1 for item in query_constraints)
    assert any(getattr(item, "max_length", None) == 2_000 for item in query_constraints)
    assert any(getattr(item, "ge", None) == 1 for item in top_k_constraints)
    assert any(getattr(item, "le", None) == 100 for item in top_k_constraints)


def test_serve_help_documents_localhost_default() -> None:
    result = CliRunner().invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "127.0.0.1" in result.output
