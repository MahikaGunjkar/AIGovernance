"""
run for initial RAG of new document

pre:  cfg has chunk/embed/vector_store sections (see config.yaml)
post: every PDF in corpus_dir is represented in the returned store, keyed by
      doc_id; documents the store already has (per store.has_doc) are
      skipped, not re-ingested
"""
from __future__ import annotations

from pathlib import Path

from heinzy.ingest import chunk, embed, extract, index, registry, structure, verify
from heinzy.ingest.types import ChunkConfig, EmbedConfig, IndexConfig, VerifyConfig
from heinzy.retrieval.store import StoredChunk, VectorStore, get_store

DEFAULT_CORPUS_DIR = Path("data/corpus")


def build_store(cfg) -> VectorStore:
    return get_store(
        cfg.vector_store.backend,
        persist_dir=getattr(cfg.vector_store, "persist_dir", None),
        host=getattr(cfg.vector_store, "host", None) or None,
        port=int(getattr(cfg.vector_store, "port", 8000) or 8000),
        collection=getattr(cfg.vector_store, "collection", None) or "heinzy",
    )


def ingest_and_populate_store(cfg, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> VectorStore:
    manifest = registry.register_corpus(corpus_dir, Path("data/index/manifest.json"))
    store = build_store(cfg)

    for entry in manifest.entries.values():
        if store.has_doc(entry.doc_id):
            print(f"already indexed, skipping ingest: {entry.pdf_path.name}")
            continue

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
