"""
Query/text embedder for retrieval.

Primary path: a local sentence-transformers model (config: embed.model_tag),
so nothing leaves local infra (PRD Data/Compute constraint). If the model or
its dependency isn't installed, we fall back to a deterministic hash embedder
of the SAME dimension so the pipeline still runs and tests stay green. The
fallback is NOT semantically meaningful — it exists only to keep the seams
wired while GPUs/models are being provisioned.

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
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_tag)
            self.is_semantic = True
        except Exception:
            # No model available -> deterministic fallback. Keeps dim invariant.
            self._model = None
            self.is_semantic = False

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return [float(x) for x in vec]
        return self._hash_embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
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
