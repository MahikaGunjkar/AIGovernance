"""
pre: doc_id is set and pdf_path exists
post: len(result) == true page_count from the PDF; all result.doc_id == doc_id
invariant: pdf_page_index is 1:1 with real PDF pages.
"""
from __future__ import annotations

from pathlib import Path

from .types import ExtractConfig, PageRecord


def extract_pages(doc_id: str, pdf_path: Path, cfg: ExtractConfig) -> list[PageRecord]:
    assert doc_id and pdf_path.exists(), "doc_id must be set and pdf_path must exist"

    raise NotImplementedError
