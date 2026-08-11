"""
Query/text embedder for retrieval.

Primary path: fastembed, loading the SAME model_tag used to embed the stored
chunks in ingest M4 (heinzy/ingest/embed.py) — so query vectors and stored
vectors come from the exact same library and land in the same space, which
cosine similarity in the store depends on. (Switched from sentence-transformers:
that library would load the same model *name* but isn't guaranteed to produce
identical vectors, which would silently degrade retrieval quality rather than
fail loudly.) If fastembed or its model isn't available, we fall back to a
deterministic hash embedder of the SAME dimension so the pipeline still runs
and tests stay green. The fallback is NOT semantically meaningful — it exists
only to keep the seams wired while GPUs/models are being provisioned.

Which path ran is exposed via `.is_semantic` and stamped by callers into logs,
so nobody mistakes fallback scores for real ones.
"""
from __future__ import annotations

import hashlib


class Embedder:
    def __init__(self, model_tag: str, dimension: int) -> None:
        self.model_tag = model_tag
        self.dimension = dimension
        self._model = None
        self.is_semantic = False
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=model_tag)
            self.is_semantic = True
        except Exception:
            # No model available -> deterministic fallback. Keeps dim invariant.
            self._model = None
            self.is_semantic = False

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            vec = next(self._model.embed([text]))
            return vec.tolist()
        return self._hash_embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return [v.tolist() for v in self._model.embed(texts)]
        return [self.embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding: repeatable across machines."""
        out: list[float] = []
        counter = 0
        while len(out) < self.dimension:
            h = hashlib.sha256(f"{text}|{counter}".encode()).digest()
            for b in h:
                if len(out) >= self.dimension:
                    break
                out.append((b / 255.0) * 2.0 - 1.0)  # map byte -> [-1, 1]
            counter += 1
        return out
