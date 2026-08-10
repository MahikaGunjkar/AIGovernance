"""
Vector-store factory / adapter tests (S4).
"""
from __future__ import annotations

import os

import pytest

from heinzy.retrieval.store import InMemoryStore, StoredChunk, get_store


def test_get_store_memory():
    store = get_store("memory")
    assert isinstance(store, InMemoryStore)
    assert store.count() == 0


def test_get_store_unknown_backend():
    with pytest.raises(NotImplementedError):
        get_store("nope")


def test_memory_add_query_roundtrip():
    store = get_store("memory")
    store.add(
        [
            StoredChunk(
                chunk_id="c1",
                vector=[1.0, 0.0, 0.0],
                text="hello",
                doc_id="d1",
                section_path="S",
                source_pages=[1],
            )
        ]
    )
    hits = store.query([1.0, 0.0, 0.0], k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"
    assert hits[0].doc_id == "d1"
    assert hits[0].source_pages == [1]


def test_chroma_requires_host():
    with pytest.raises(ValueError, match="CHROMA_HOST"):
        get_store("chroma", host="", port=8000, collection="heinzy")


def test_chroma_missing_package_or_unreachable():
    # Without a live server (or without chromadb installed), construction fails
    # with a clear error — never a silent empty store.
    with pytest.raises((ImportError, ConnectionError, ValueError)):
        get_store("chroma", host="127.0.0.1", port=59999, collection="heinzy-test")


@pytest.mark.integration
def test_chroma_live_roundtrip():
    host = os.environ.get("CHROMA_HOST", "").strip()
    if not host:
        pytest.skip("CHROMA_HOST not set")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    try:
        store = get_store(
            "chroma",
            host=host,
            port=port,
            collection="heinzy-pytest",
        )
    except (ImportError, ConnectionError) as exc:
        pytest.skip(str(exc))

    store.add(
        [
            StoredChunk(
                chunk_id="pytest-c1",
                vector=[1.0, 0.0, 0.0] + [0.0] * 381,
                text="pytest chunk",
                doc_id="pytest-doc",
                section_path="T > S",
                source_pages=[9],
            )
        ]
    )
    assert store.count() >= 1
    hits = store.query([1.0, 0.0, 0.0] + [0.0] * 381, k=1)
    assert hits
    assert hits[0].doc_id == "pytest-doc"
    assert hits[0].source_pages == [9]
