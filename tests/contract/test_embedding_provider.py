"""Contract every EmbeddingProvider implementation must pass, plus the prefix contract.

Real models run only where they are downloaded (the Nano); elsewhere those
cases are skipped and only the Fake implementation is checked.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pytest

from memrag.infrastructure.embedding.fake import FakeEmbeddingProvider
from memrag.infrastructure.embedding.registry import load_registry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = load_registry(ROOT / "configs" / "models.yaml", ROOT)

# Prefixes copied from each model card (checked 2026-11-07). A change to
# configs/models.yaml that does not match the model card fails here.
OFFICIAL_PREFIXES = {
    "cl-nagoya/ruri-v3-30m": ("検索クエリ: ", "検索文書: "),
    "cl-nagoya/ruri-v3-130m": ("検索クエリ: ", "検索文書: "),
    "cl-nagoya/ruri-v3-310m": ("検索クエリ: ", "検索文書: "),
    "Qwen/Qwen3-Embedding-0.6B": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", ""),
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "BAAI/bge-base-zh-v1.5": ("为这个句子生成表示以用于检索相关文章：", ""),
}

TEXTS = ["田中さんはラーメンが好き。", "佐藤さんは来月、京都へ旅行する予定。"]
SPECS = list(REGISTRY.models.values())


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_prefix_matches_model_card(spec):
    assert spec.hf_id in OFFICIAL_PREFIXES, f"add the model card prefixes of {spec.hf_id} to this test"
    assert (spec.query_prefix, spec.document_prefix) == OFFICIAL_PREFIXES[spec.hf_id]


@pytest.mark.models
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_prefix_matches_builtin_prompts(spec):
    """When a model ships Sentence Transformers prompts, ours must be identical."""
    path = REGISTRY.local_path(spec)
    if path is None:
        pytest.skip("model not downloaded")
    cfg = path / "config_sentence_transformers.json"
    prompts = json.loads(cfg.read_text(encoding="utf-8")).get("prompts", {}) if cfg.exists() else {}
    if "query" in prompts:
        assert spec.query_prefix == prompts["query"]
    if "document" in prompts:
        assert spec.document_prefix == prompts["document"]


def _fake(spec):
    return FakeEmbeddingProvider(spec)


def _real(spec):
    path = REGISTRY.local_path(spec)
    if path is None:
        pytest.skip("model not downloaded")
    import torch

    from memrag.infrastructure.embedding.sentence_transformer import SentenceTransformerProvider

    return SentenceTransformerProvider(spec, path, device="cuda" if torch.cuda.is_available() else "cpu")


PROVIDERS = [pytest.param((_fake, s), id=f"fake-{s.name}") for s in SPECS] + [
    pytest.param((_real, s), id=f"real-{s.name}", marks=pytest.mark.models) for s in SPECS
]


@pytest.fixture(params=PROVIDERS)
def provider(request):
    factory, spec = request.param
    p = factory(spec)
    yield p
    del p
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def test_info_returns_registry_spec(provider):
    assert provider.info() == REGISTRY.get(provider.info().name)


def test_explain_shows_exact_encoded_text(provider):
    spec = provider.info()
    assert provider.explain(TEXTS[0], "query").encoded_text == spec.query_prefix + TEXTS[0]
    assert provider.explain(TEXTS[0], "document").encoded_text == spec.document_prefix + TEXTS[0]


def test_output_shape_and_unit_norm(provider):
    spec = provider.info()
    for vecs in (provider.embed_queries(TEXTS), provider.embed_documents(TEXTS)):
        arr = np.asarray(vecs, dtype=np.float32)
        assert arr.shape == (len(TEXTS), spec.dim)
        assert np.all(np.isfinite(arr))
        np.testing.assert_allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-3)


def test_same_input_gives_same_vector(provider):
    a = np.asarray(provider.embed_queries(TEXTS[:1]))
    b = np.asarray(provider.embed_queries(TEXTS[:1]))
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_batch_matches_single(provider):
    batch = np.asarray(provider.embed_documents(TEXTS))
    single = np.asarray(provider.embed_documents(TEXTS[1:]))
    np.testing.assert_allclose(batch[1], single[0], atol=2e-2)


def test_query_and_document_differ_only_by_prefix(provider):
    spec = provider.info()
    q = np.asarray(provider.embed_queries(TEXTS[:1]))[0]
    d = np.asarray(provider.embed_documents(TEXTS[:1]))[0]
    if spec.query_prefix == spec.document_prefix:
        np.testing.assert_allclose(q, d, atol=1e-5)
    else:
        assert not np.allclose(q, d, atol=1e-4), "query and document prefixes are not being applied"


@pytest.mark.models
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_provider_applies_prefix_before_model(spec):
    """embed_queries(t) must equal the raw model's encoding of prefix + t."""
    p = _real(spec)
    ours = np.asarray(p.embed_queries(TEXTS[:1]))[0]
    raw = p.model.encode([spec.query_prefix + TEXTS[0]], convert_to_numpy=True)[0].astype(np.float32)
    np.testing.assert_allclose(ours, raw / np.linalg.norm(raw), atol=1e-4)
    del p
    gc.collect()
