"""
pre: pages are sorted and contiguous by pdf_page_index
post: all blocks have section_path set (or "unresolved"); block text is a
    substring of its source page
invariant: section_path is non-null or explicitly "unresolved".
"""
from __future__ import annotations

from .types import Block, PageRecord, StructureConfig


def build_sections(pages: list[PageRecord], cfg: StructureConfig) -> list[Block]:
    indices = [p.pdf_page_index for p in pages]
    assert indices == sorted(indices) and indices == list(range(indices[0], indices[0] + len(indices))), \
        "pages must be sorted and contiguous by pdf_page_index"

    raise NotImplementedError
