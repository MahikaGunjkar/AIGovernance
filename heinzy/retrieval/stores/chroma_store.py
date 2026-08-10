"""
Chroma HTTP adapter (Infra task S4).

Talks to a shared Chroma server via HttpClient. Embeddings are supplied by
Heinzy (hash fallback or BGE) — Chroma only stores/searches vectors.

pre:  host/port reachable; chromadb optional extra installed
post: add/query/count match the VectorStore protocol; provenance round-trips
invariant: retrieval code never imports chromadb — only this adapter does.
"""
from __future__ import annotations

import json
from typing import Any

from heinzy.retrieval.store import ScoredChunk, StoredChunk


class ChromaStore:
    """VectorStore backed by a remote Chroma collection (cosine space)."""

    def __init__(
        self,
        host: str,
        port: int = 8000,
        collection: str = "heinzy",
        **_: Any,
    ) -> None:
        if not host or not str(host).strip():
            raise ValueError(
                "vector_store.host / CHROMA_HOST must be set when backend=chroma"
            )
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "chromadb-client is required for backend=chroma. "
                'Install with: pip install -e ".[store]" '
                "(thin HTTP client — works on Mac and Windows without C++ build tools)."
            ) from exc

        self.host = str(host).strip()
        self.port = int(port)
        self.collection_name = collection or "heinzy"
        try:
            self._client = chromadb.HttpClient(host=self.host, port=self.port)
            self._client.heartbeat()
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach Chroma at {self.host}:{self.port}. "
                "Is docker-compose.chroma.yml up, and is the firewall open?"
            ) from exc
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[StoredChunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        embeddings = [c.vector for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [_meta_from_chunk(c) for c in chunks]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, vector: list[float], k: int) -> list[ScoredChunk]:
        if k <= 0:
            return []
        n = min(k, max(self.count(), 0))
        if n == 0:
            return []
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        hits: list[ScoredChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            # Cosine space: Chroma distance ~= 1 - cosine_similarity.
            score = 1.0 - float(dist)
            hits.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    score=score,
                    doc_id=str(meta.get("doc_id", "")),
                    section_path=meta.get("section_path"),
                    source_pages=_parse_pages(meta.get("source_pages")),
                )
            )
        hits.sort(key=lambda s: (-s.score, s.chunk_id))
        return hits

    def count(self) -> int:
        return int(self._collection.count())


def _meta_from_chunk(chunk: StoredChunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "section_path": chunk.section_path if chunk.section_path is not None else "",
        "source_pages": json.dumps(list(chunk.source_pages)),
    }


def _parse_pages(raw: Any) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return []
