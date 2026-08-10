"""
pre: pdf_dir exists and contains >=1 *.pdf
post: len(result.entries) == number of PDFs in pdf_dir
invariant: sha256 -> doc_id is deterministic across machines.
"""
from __future__ import annotations

import pymupdf as mu
from pathlib import Path
import hashlib
from .types import Manifest, DocEntry


def register_corpus(pdf_dir: Path, manifest_path: Path) -> Manifest:
    assert pdf_dir.is_dir() and any(pdf_dir.glob("*.pdf")), \
        "pdf_dir must exist and contain >=1 *.pdf"

    entries = {}
    for pdf_path in pdf_dir.glob("*.pdf"):
        digest = _hash_file(pdf_path)
        pages = _page_count(pdf_path)
        entries[digest] = DocEntry(doc_id= digest, sha256=digest,
                                   pdf_path=pdf_path, page_count=pages)

    return Manifest(entries=entries)


def _hash_file(path):
    BUF_SIZE = 65536
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _page_count(path):
    doc = mu.open(path)
    doc_len = len(doc)
    doc.close()
    return doc_len
