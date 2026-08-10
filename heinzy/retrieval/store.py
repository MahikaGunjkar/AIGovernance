"""
Vector-store adapter (Infra task S4).

This is the seam between retrieval and whatever DB the team lands on. Retrieval
code depends ONLY on the VectorStore protocol below — never on a concrete store.
To add a real DB (Chroma, pgvector, etc.), the DB owner writes one class that
implements this protocol and registers it in `get_store()`. No retrieval code
changes.

Tonight's default is InMemoryStore: brute-force cosine over a Python list. Zero
infra, good enough to prove the pipeline end to end and to run eval on a small
corpus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieval hit: the chunk plus its similarity score and provenance.

    Provenance fields (doc_id, section_path, source_pages) are what make
    citations verifiable downstream (task A4) — they travel with every hit.
    """

    chunk_id: str
    text: str
    score: float
    doc_id: str
    section_path: str | None
    source_pages: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class StoredChunk:
    """What we put INTO the store: a vector plus the metadata to cite it later."""

    chunk_id: str
    vector: list[float]
    text: str
    doc_id: str
    section_path: str | None
    source_pages: list[int] = field(default_factory=list)


class VectorStore(Protocol):
    """The only interface retrieval knows about."""

    def add(self, chunks: list[StoredChunk]) -> None: ...

    def query(self, vector: list[float], k: int) -> list[ScoredChunk]: ...

    def count(self) -> int: ...

    def has_doc(self, doc_id: str) -> bool: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryStore:
    """Brute-force cosine store. Deterministic, dependency-free, swap-later."""

    def __init__(self) -> None:
        self._items: list[StoredChunk] = []

    def add(self, chunks: list[StoredChunk]) -> None:
        self._items.extend(chunks)

    def query(self, vector: list[float], k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(
                chunk_id=it.chunk_id,
                text=it.text,
                score=_cosine(vector, it.vector),
                doc_id=it.doc_id,
                section_path=it.section_path,
                source_pages=it.source_pages,
            )
            for it in self._items
        ]
        # Sort by score desc; tie-break on chunk_id for stable, reproducible order.
        scored.sort(key=lambda s: (-s.score, s.chunk_id))
        return scored[:k]

    def count(self) -> int:
        return len(self._items)

    def has_doc(self, doc_id: str) -> bool:
        return any(it.doc_id == doc_id for it in self._items)


def get_store(backend: str, **kwargs) -> VectorStore:
    """Factory. Reads vector_store.backend from config.

    Supported:
      - memory: in-process list (tests / offline)
      - chroma: shared Chroma HTTP server (see docker-compose.chroma.yml)
    """
    if backend == "memory":
        return InMemoryStore()
    if backend == "chroma":
        from heinzy.retrieval.stores.chroma_store import ChromaStore

        return ChromaStore(**kwargs)
    raise NotImplementedError(
        f"vector_store.backend={backend!r} has no adapter yet. "
        "Implement the VectorStore protocol and register it in get_store()."
    )
