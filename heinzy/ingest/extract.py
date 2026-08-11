"""
#Extract page info to create a pageRecord obejct for each page
pre: doc_id is set and pdf_path exists
post: len(result) == true page_count from the PDF; all result.doc_id == doc_id
invariant: pdf_page_index is 1:1 with real PDF pages.
"""
from __future__ import annotations

from pathlib import Path
import pymupdf4llm

from .types import ExtractConfig, PageRecord

def extract_pages(doc_id: str, pdf_path: Path, cfg: ExtractConfig) -> list[PageRecord]:
    assert doc_id and pdf_path.exists()

    page_dicts = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages = []

    for page_dict in page_dicts:
        pages.append(
            PageRecord(
            doc_id= doc_id,
            pdf_page_index= page_dict["metadata"]["page_number"],
            text= page_dict["text"]
        ))

    return pages

