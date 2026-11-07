"""memrag: Japanese conversational-memory RAG, embedding model evaluation."""

from memrag.api import (
    MemRAG,
    annotation_agreement,
    assign_dataset_splits,
    check_environment,
    download_models,
    evaluate_retrieval,
    list_models,
    load_gold_set,
    make_second_pass_sample,
    probe_models,
    show_query,
    validate_dataset,
)
from memrag.domain.errors import ConfigError, MemRAGError, ModelNotAvailableError
from memrag.domain.schemas.embedding import EncodedText, ModelSpec

__all__ = [
    "MemRAG",
    "annotation_agreement",
    "assign_dataset_splits",
    "check_environment",
    "download_models",
    "evaluate_retrieval",
    "list_models",
    "load_gold_set",
    "make_second_pass_sample",
    "probe_models",
    "show_query",
    "validate_dataset",
    "ConfigError",
    "MemRAGError",
    "ModelNotAvailableError",
    "EncodedText",
    "ModelSpec",
]
