"""
Retrieval (Prototype task A2).

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

from dataclasses import dataclass, field
from typing import Any

from heinzy.common.config import Config
from heinzy.eventlog.actor import Actor
from heinzy.eventlog.writer import JsonlEventLog
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
    # Set when an event log was attached; the full A5 envelope (id/ts/type/actor + payload).
    audit_record: dict[str, Any] | None = field(default=None, repr=False)

    def to_log_record(self) -> dict:
        """Retrieval payload for the JSON event log (task A5)."""
        return {
            "query": self.query,
            "k": self.k,
            "embed_model": self.embed_model,
            "is_semantic": self.is_semantic,
            "config_hash": self.config_hash,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "score": round(h.score, 6),
                    "doc_id": h.doc_id,
                    "section_path": h.section_path,
                    "source_pages": h.source_pages,
                }
                for h in self.hits
            ],
        }


class Retriever:
    """Thin, config-driven retrieval front door.

    Everything tunable (k, score_floor, embedder, store backend) comes from the
    Config object. Construct once, call `retrieve()` per question.

    Pass an EventLog to persist A5 audit records on each successful retrieve.
    When logging is enabled, an Actor is required (per-call or default_actor).
    Tests omit the event log so they stay filesystem-free.
    """

    def __init__(
        self,
        cfg: Config,
        store: VectorStore | None = None,
        event_log: JsonlEventLog | None = None,
        default_actor: Actor | None = None,
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
        self.event_log = event_log
        self.default_actor = default_actor

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        actor: Actor | None = None,
    ) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = k if k is not None else self.cfg.retrieval.k
        score_floor = getattr(self.cfg.retrieval, "score_floor", 0.0)

        qvec = self.embedder.embed(query)
        hits = self.store.query(qvec, k=k)
        if score_floor > 0:
            hits = [h for h in hits if h.score >= score_floor]

        result = RetrievalResult(
            query=query,
            hits=hits,
            k=k,
            embed_model=self.cfg.embed.model_tag,
            is_semantic=self.embedder.is_semantic,
            config_hash=self.cfg.config_hash,
        )
        if self.event_log is not None:
            resolved = actor if actor is not None else self.default_actor
            if resolved is None:
                raise ValueError(
                    "actor is required when event logging is enabled "
                    "(pass actor= to retrieve(), or default_actor= to Retriever)"
                )
            result.audit_record = self.event_log.append_retrieval(result, actor=resolved)
        return result