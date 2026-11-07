"""ChromaDB installs and passes a read/write round trip with a metadata filter."""

from memrag.infrastructure.environment import check_chromadb


def test_chromadb_read_write_with_filter():
    ok, detail = check_chromadb()
    assert ok, detail
    assert detail["filtered_query_ids"] == ["m1", "m3"]
