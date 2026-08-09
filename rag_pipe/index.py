"""
pre: every EmbeddedChunk has a matching Chunk record
post: result.name == result.config_hash
invariant: re-running identical config on identical corpus is a no-op.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .types import Chunk, Collection, EmbeddedChunk, IndexConfig

_INDEX_DIR = Path("rag_pipe/.index")


def build_index(embedded: list[EmbeddedChunk], chunks: list[Chunk], cfg: IndexConfig) -> Collection:
    chunk_ids = {c.chunk_id for c in chunks}
    assert all(e.chunk_id in chunk_ids for e in embedded), \
        "every EmbeddedChunk must have a matching Chunk record"

    model_tags = {e.model_tag for e in embedded}
    fingerprint = ",".join(sorted(e.chunk_id for e in embedded)) + "|" + ",".join(sorted(model_tags))
    config_hash = hashlib.sha256(fingerprint.encode()).hexdigest()

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _INDEX_DIR / f"{config_hash}.json"

    if not out_path.exists():
        data = [
            {"chunk_id": e.chunk_id, "vector": e.vector, "model_tag": e.model_tag}
            for e in embedded
        ]
        out_path.write_text(json.dumps(data))

    return Collection(name=config_hash, config_hash=config_hash)
