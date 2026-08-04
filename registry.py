"""
pre: pdf_dir exists and contains >=1 *.pdf
post: len(result.entries) == number of PDFs in pdf_dir
invariant: sha256 -> doc_id is deterministic across machines.
"""
from __future__ import annotations

from pathlib import Path

from .types import Manifest


def register_corpus(pdf_dir: Path, manifest_path: Path) -> Manifest:
    assert pdf_dir.is_dir() and any(pdf_dir.glob("*.pdf")), \
        "pdf_dir must exist and contain >=1 *.pdf"

    raise NotImplementedError
