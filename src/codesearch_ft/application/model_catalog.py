"""Catalog of supported model identifiers without network or cache access."""

from codesearch_ft.domain.models import ConfiguredModel

_SUPPORTED_MODELS = (
    ConfiguredModel(
        model_id="microsoft/codebert-base",
        role="shared query/code encoder",
        source="Hugging Face model identifier",
    ),
)


def list_configured_models() -> tuple[ConfiguredModel, ...]:
    """Return configured model metadata; do not inspect caches or download weights."""
    return _SUPPORTED_MODELS
