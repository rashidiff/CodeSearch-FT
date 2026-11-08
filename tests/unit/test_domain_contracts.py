"""Smoke checks for framework-independent package contracts."""

from codesearch_ft.domain.models import ConfiguredModel, RetrievalMode
from codesearch_ft.domain.ports import (
    CodeEncoder,
    CodeParser,
    ExperimentTracker,
    LexicalRetriever,
    MetadataRepository,
    ModelCatalog,
    SearchService,
    SemanticRetriever,
    VectorIndex,
)


def test_retrieval_modes_and_configured_model_contract() -> None:
    model = ConfiguredModel(
        model_id="microsoft/codebert-base",
        role="shared query/code encoder",
        source="Hugging Face model identifier",
    )

    assert [mode.value for mode in RetrievalMode] == ["lexical", "semantic", "hybrid"]
    assert model.weights_checked is False


def test_required_adapter_protocols_are_importable() -> None:
    assert callable(CodeParser.extract)
    assert callable(CodeEncoder.encode_queries)
    assert callable(MetadataRepository.replace_file_functions)
    assert callable(VectorIndex.search)
    assert callable(LexicalRetriever.search)
    assert callable(SemanticRetriever.search)
    assert callable(SearchService.search)
    assert callable(ExperimentTracker.log_metrics)
    assert callable(ModelCatalog.list_models)
