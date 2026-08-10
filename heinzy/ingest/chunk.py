"""
pre: cfg.target_tokens > cfg.overlap_tokens >= 0
post: chunk text is substring-preserved from its source block; no chunk
    crosses a section boundary; table chunks stay atomic
invariant: no chunk crosses a section boundary; tables stay atomic.
"""
from __future__ import annotations

import tiktoken

from .types import Block, Chunk, ChunkConfig

_enc = tiktoken.get_encoding("cl100k_base")


def _is_table(text):
    return "|" in text


def chunk_blocks(blocks: list[Block], cfg: ChunkConfig) -> list[Chunk]:
    assert cfg.target_tokens > cfg.overlap_tokens >= 0, \
        "target_tokens must be > overlap_tokens >= 0"

    chunks = []
    n = 0
    step = cfg.target_tokens - cfg.overlap_tokens

    for block in blocks:
        if _is_table(block.text):
            chunks.append(Chunk(
                doc_id=block.doc_id,
                chunk_id=f"{block.doc_id}-{n}",
                text=block.text,
                section_path=block.section_path,
                is_table=True,
            ))
            n += 1
            continue

        tokens = _enc.encode(block.text)
        i = 0
        while i < len(tokens):
            window = tokens[i:i + cfg.target_tokens]
            chunks.append(Chunk(
                doc_id=block.doc_id,
                chunk_id=f"{block.doc_id}-{n}",
                text=_enc.decode(window),
                section_path=block.section_path,
                is_table=False,
            ))
            n += 1
            if len(window) < cfg.target_tokens:
                break
            i += step

    return chunks
