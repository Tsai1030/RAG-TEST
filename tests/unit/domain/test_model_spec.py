import pytest

from memrag.domain.schemas.embedding import ModelSpec


def make_spec(**overrides) -> ModelSpec:
    values = dict(
        name="m", hf_id="org/m", revision="0" * 40, role="candidate", dim=4, max_seq_length=512,
        query_prefix="検索クエリ: ", document_prefix="検索文書: ", dtype="float32",
        trust_remote_code=False, license="MIT", commercial_ok=True,
    )
    return ModelSpec(**(values | overrides))


def test_encode_text_adds_query_prefix():
    enc = make_spec().encode_text("田中さんの好物は？", "query")
    assert enc.prefix == "検索クエリ: "
    assert enc.encoded_text == "検索クエリ: 田中さんの好物は？"
    assert enc.text == "田中さんの好物は？"


def test_encode_text_adds_document_prefix():
    assert make_spec().encode_text("寿司が好き", "document").encoded_text == "検索文書: 寿司が好き"


def test_empty_prefix_leaves_text_unchanged():
    assert make_spec(document_prefix="").encode_text("寿司が好き", "document").encoded_text == "寿司が好き"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        make_spec().prefix("passage")
