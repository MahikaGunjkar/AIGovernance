"""
Retrieval tests (A2 "done when": k and chunk size change without editing source).

These use the injected in-memory store so they run fast and deterministically,
with no model download and no DB. They lock the retrieval *contract*, which is
what protects you when someone swaps the store later.
"""
from __future__ import annotations

import pytest

from heinzy.common.config import load_config
from heinzy.retrieval.embedder import Embedder
from heinzy.retrieval.retrieve import Retriever
from heinzy.retrieval.store import InMemoryStore, StoredChunk

CHUNKS = [
    ("c1", "graduation requires 144 units", "H > Grad", [1]),
    ("c2", "core includes statistics", "H > Core", [2]),
    ("c3", "up to 4 electives allowed", "H > Electives", [3]),
]


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def seeded_store(cfg):
    emb = Embedder(cfg.embed.model_tag, cfg.embed.dimension)
    store = InMemoryStore()
    store.add([
        StoredChunk(cid, emb.embed(text), text, "doc-x", sec, pages)
        for cid, text, sec, pages in CHUNKS
    ])
    return store


def test_returns_at_most_k(cfg, seeded_store):
    r = Retriever(cfg, store=seeded_store)
    result = r.retrieve("anything", k=2)
    assert len(result.hits) <= 2


def test_k_is_config_driven(cfg, seeded_store):
    # No k passed -> uses cfg.retrieval.k. Proves behavior comes from config.
    r = Retriever(cfg, store=seeded_store)
    result = r.retrieve("anything")
    assert result.k == cfg.retrieval.k


def test_hits_sorted_descending(cfg, seeded_store):
    r = Retriever(cfg, store=seeded_store)
    hits = r.retrieve("statistics core", k=3).hits
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_provenance_present(cfg, seeded_store):
    # Every hit must carry citation fields (feeds task A4).
    r = Retriever(cfg, store=seeded_store)
    for h in r.retrieve("units", k=3).hits:
        assert h.doc_id
        assert h.section_path is not None
        assert isinstance(h.source_pages, list)


def test_empty_query_rejected(cfg, seeded_store):
    r = Retriever(cfg, store=seeded_store)
    with pytest.raises(ValueError):
        r.retrieve("   ")


def test_log_record_shape(cfg, seeded_store):
    r = Retriever(cfg, store=seeded_store)
    rec = r.retrieve("units", k=2).to_log_record()
    assert set(rec) >= {"query", "k", "embed_model", "config_hash", "hits"}
    assert rec["config_hash"] == cfg.config_hash
