"""
Ingest the real MISM handbook (data/corpus/) and populate the retrieval store,
bridging M0-M6 ingest output into the A2/S4 retrieval layer with real data.
The other smoke scripts prove retrieval works against invented placeholder
chunks; this one proves it end to end against the actual handbook.

Run from repo root:
    python scripts/ingest_and_retrieve_demo.py
    python scripts/ingest_and_retrieve_demo.py --query "how many electives"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from heinzy.common.config import load_config
from heinzy.ingest import chunk, embed, extract, index, registry, structure, verify
from heinzy.ingest.types import ChunkConfig, EmbedConfig, IndexConfig, VerifyConfig
from heinzy.retrieval.retrieve import Retriever
from heinzy.retrieval.store import StoredChunk, get_store

CORPUS_DIR = Path("data/corpus")


def ingest_and_populate_store(cfg):
    manifest = registry.register_corpus(CORPUS_DIR, Path("data/index/manifest.json"))

    store = get_store(
        cfg.vector_store.backend,
        persist_dir=getattr(cfg.vector_store, "persist_dir", None),
        host=getattr(cfg.vector_store, "host", None) or None,
        port=int(getattr(cfg.vector_store, "port", 8000) or 8000),
        collection=getattr(cfg.vector_store, "collection", None) or "heinzy",
    )

    for entry in manifest.entries.values():
        pages = extract.extract_pages(entry.doc_id, entry.pdf_path, None)
        blocks = structure.build_sections(pages, None)
        chunks = chunk.chunk_blocks(
            blocks,
            ChunkConfig(
                target_tokens=cfg.chunk.target_tokens,
                overlap_tokens=cfg.chunk.overlap_tokens,
            ),
        )
        embedded = embed.embed_chunks(
            chunks,
            EmbedConfig(model_tag=cfg.embed.model_tag, dimension=cfg.embed.dimension),
        )
        index.build_index(embedded, chunks, IndexConfig())
        report = verify.verify(chunks, pages, VerifyConfig())
        if not report.passed:
            print(f"verify FAILED for {entry.pdf_path.name}: {report.failures}")

        # source_pages lives on Block, not Chunk -- recover it via section_path
        pages_by_section = {b.section_path: b.source_pages for b in blocks}
        vectors_by_id = {e.chunk_id: e.vector for e in embedded}

        store.add([
            StoredChunk(
                chunk_id=c.chunk_id,
                vector=vectors_by_id[c.chunk_id],
                text=c.text,
                doc_id=c.doc_id,
                section_path=c.section_path,
                source_pages=pages_by_section.get(c.section_path, []),
            )
            for c in chunks
        ])
        print(f"ingested {entry.pdf_path.name}: {len(pages)} pages -> {len(chunks)} chunks")

    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="What are the required courses?")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    store = ingest_and_populate_store(cfg)
    print(f"\nstore populated: {store.count()} chunks\n")

    retriever = Retriever(cfg, store=store)
    result = retriever.retrieve(args.query, k=args.k)

    print(f"query: {result.query}")
    print(f"embedder: {result.embed_model} "
          f"({'semantic' if result.is_semantic else 'HASH-FALLBACK, not semantic'})\n")
    for rank, h in enumerate(result.hits, 1):
        print(f"  {rank}. score={h.score:.4f}  [{h.section_path}]  p{h.source_pages}")
        print(f"     {h.text[:150]}...")


if __name__ == "__main__":
    main()
