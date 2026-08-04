"""
pre: cfg.target_tokens > cfg.overlap_tokens >= 0
post: chunk text is substring-preserved from its source block; no chunk
    crosses a section boundary; table chunks stay atomic
invariant: no chunk crosses a section boundary; tables stay atomic.
"""
from __future__ import annotations

from .types import Block, Chunk, ChunkConfig


def chunk_blocks(blocks: list[Block], cfg: ChunkConfig) -> list[Chunk]:
    assert cfg.target_tokens > cfg.overlap_tokens >= 0, \
        "target_tokens must be > overlap_tokens >= 0"

    raise NotImplementedError
