"""
pre: every EmbeddedChunk has a matching Chunk record
post: result.name == result.config_hash
invariant: re-running identical config on identical corpus is a no-op.
"""
from __future__ import annotations

from .types import Chunk, Collection, EmbeddedChunk, IndexConfig


def build_index(embedded: list[EmbeddedChunk], chunks: list[Chunk], cfg: IndexConfig) -> Collection:
    chunk_ids = {c.chunk_id for c in chunks}
    assert all(e.chunk_id in chunk_ids for e in embedded), \
        "every EmbeddedChunk must have a matching Chunk record"

    raise NotImplementedError
