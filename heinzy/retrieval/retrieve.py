"""
pre:  the store has been populated (ingest pipeline, or a test fixture)
post: query() returns <= k ScoredChunks, sorted by score descending, each
      carrying provenance (doc_id, section_path, source_pages) for citations
invariant: k and the embedding model are read from config — never hardcoded.
           Changing retrieval behavior means editing config.yaml, not source.

Design note: this module is deliberately store-agnostic. It talks to the
VectorStore protocol only, so the team can swap the in-memory store for a real
DB without touching this file (S4).
"""
from __future__ import annotations

from dataclasses import dataclass

from heinzy.common.config import Config
from heinzy.retrieval.embedder import Embedder
from heinzy.retrieval.store import ScoredChunk, VectorStore, get_store


@dataclass
class RetrievalResult:
    query: str
    hits: list[ScoredChunk]
    k: int
    embed_model: str
    is_semantic: bool
    config_hash: str


class Retriever:
    """Thin, config-driven retrieval front door.

    Everything tunable (k, score_floor, embedder, store backend) comes from the
    Config object. Construct once, call `retrieve()` per question.
    """

    def __init__(
        self,
        cfg: Config,
        store: VectorStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.embedder = Embedder(
            model_tag=cfg.embed.model_tag,
            dimension=cfg.embed.dimension,
        )
        # Use an injected store (tests) or build one from config (real runs).
        vs = cfg.vector_store
        self.store = store or get_store(
            vs.backend,
            persist_dir=getattr(vs, "persist_dir", None),
            host=getattr(vs, "host", None) or None,
            port=int(getattr(vs, "port", 8000) or 8000),
            collection=getattr(vs, "collection", None) or "heinzy",
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = k if k is not None else self.cfg.retrieval.k
        score_floor = getattr(self.cfg.retrieval, "score_floor", 0.0)

        qvec = self.embedder.embed(query)
        hits = self.store.query(qvec, k=k)
        if score_floor > 0:
            hits = [h for h in hits if h.score >= score_floor]

        return RetrievalResult(
            query=query,
            hits=hits,
            k=k,
            embed_model=self.cfg.embed.model_tag,
            is_semantic=self.embedder.is_semantic,
            config_hash=self.cfg.config_hash,
        )
