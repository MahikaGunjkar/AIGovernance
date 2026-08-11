"""
pre: cfg.model_tag and cfg.dimension must be set
post: len(result) == len(chunks); every vector has len == cfg.dimension;
    every result.model_tag == cfg.model_tag
invariant: same text + same model_tag -> same vector, deterministic.
"""
from __future__ import annotations

from fastembed import TextEmbedding

from .types import Chunk, EmbedConfig, EmbeddedChunk


def embed_chunks(chunks: list[Chunk], cfg: EmbedConfig) -> list[EmbeddedChunk]:
    assert cfg.model_tag and cfg.dimension > 0, "cfg.model_tag and cfg.dimension must be set"

    model = TextEmbedding(model_name=cfg.model_tag)
    texts = [c.text for c in chunks]
    vectors = model.embed(texts)

    result = []
    for chunk, vector in zip(chunks, vectors):
        result.append(EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            vector=vector.tolist(),
            model_tag=cfg.model_tag,
        ))
    return result
