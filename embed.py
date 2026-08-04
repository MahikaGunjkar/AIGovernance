"""
pre: cfg.model_tag and cfg.dimension must be set
post: len(result) == len(chunks); every vector has len == cfg.dimension;
    every result.model_tag == cfg.model_tag
invariant: same text + same model_tag -> same vector, deterministic.
"""
from __future__ import annotations

from .types import Chunk, EmbedConfig, EmbeddedChunk


def embed_chunks(chunks: list[Chunk], cfg: EmbedConfig) -> list[EmbeddedChunk]:
    assert cfg.model_tag and cfg.dimension > 0, "cfg.model_tag and cfg.dimension must be set"

    raise NotImplementedError
