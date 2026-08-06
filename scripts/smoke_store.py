"""
Smoke test for the vector-store seam (S4).

Builds a store from config (memory or chroma), seeds placeholder chunks, runs
one query. Proves get_store() + adapter wiring without the full retrieval CLI.

Run from repo root (load .env into the process first if needed):
    python scripts/smoke_store.py
    # with shared Chroma:
    #   set vector_store.backend: chroma in config.yaml
    #   CHROMA_HOST=127.0.0.1  (or teammate LAN IP)
    #   pip install -e ".[store]"
"""
from __future__ import annotations

import os
from pathlib import Path

from heinzy.common.config import load_config
from heinzy.retrieval.embedder import Embedder
from heinzy.retrieval.store import StoredChunk, get_store

PLACEHOLDER = [
    ("c1", "MISM students must complete 144 units to graduate.", "MISM Handbook > Graduation", [12]),
    ("c2", "The core curriculum includes statistics, economics, and databases.", "MISM Handbook > Core", [8]),
    ("c3", "Students may take up to 4 elective courses outside Heinz.", "MISM Handbook > Electives", [21]),
]


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no dependency). Does not override existing env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> None:
    _load_dotenv()
    cfg = load_config()
    vs = cfg.vector_store
    print(f"backend     : {vs.backend}")
    print(f"config_hash : {cfg.config_hash}")
    if vs.backend == "chroma":
        print(f"chroma      : {getattr(vs, 'host', '')}:{getattr(vs, 'port', 8000)} "
              f"collection={getattr(vs, 'collection', 'heinzy')}")

    store = get_store(
        vs.backend,
        host=getattr(vs, "host", None) or None,
        port=int(getattr(vs, "port", 8000) or 8000),
        collection=getattr(vs, "collection", None) or "heinzy",
        persist_dir=getattr(vs, "persist_dir", None),
    )
    embedder = Embedder(cfg.embed.model_tag, cfg.embed.dimension)
    stored = [
        StoredChunk(
            chunk_id=cid,
            vector=embedder.embed(text),
            text=text,
            doc_id="placeholder-mism-handbook",
            section_path=section,
            source_pages=pages,
        )
        for cid, text, section, pages in PLACEHOLDER
    ]
    store.add(stored)
    qvec = embedder.embed("How many electives can I take?")
    hits = store.query(qvec, k=2)

    print(f"store.count : {store.count()}")
    print(f"query hits  : {len(hits)}")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. score={h.score:.4f}  [{h.section_path}]  {h.text[:60]}...")


if __name__ == "__main__":
    main()
