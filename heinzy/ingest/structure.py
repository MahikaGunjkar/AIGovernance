"""
pre: pages are sorted and contiguous by pdf_page_index
post: all blocks have section_path set (or "unresolved"); block text is a
    substring of its source page
invariant: section_path is non-null or explicitly "unresolved".
"""
from __future__ import annotations

from .types import Block, PageRecord, StructureConfig


def _is_heading(line):
    line = line.strip()
    stripped = line.lstrip("#")
    return stripped != line and stripped.startswith(" ")


def _heading_text(line):
    text = line.strip()
    text = text.lstrip("#")
    text = text.strip()
    text = text.strip("*")
    text = text.strip()
    return text


def build_sections(pages: list[PageRecord], cfg: StructureConfig) -> list[Block]:
    indices = [p.pdf_page_index for p in pages]
    assert indices == sorted(indices) and indices == list(range(indices[0], indices[0] + len(indices))), \
        "pages must be sorted and contiguous by pdf_page_index"

    blocks = []
    current_section = "unresolved"
    current_lines = []
    current_pages = []

    doc_id = pages[0].doc_id

    for page in pages:
        for line in page.text.splitlines():
            if _is_heading(line):
                if current_lines:
                    blocks.append(Block(
                        doc_id=doc_id,
                        text="\n".join(current_lines),
                        section_path=current_section,
                        source_pages=current_pages
                    ))
                    current_lines = []
                    current_pages = []
                current_section = _heading_text(line)

            current_lines.append(line)
            if page.pdf_page_index not in current_pages:
                current_pages.append(page.pdf_page_index)

    if current_lines:
        blocks.append(Block(
            doc_id=doc_id,
            text="\n".join(current_lines),
            section_path=current_section,
            source_pages=current_pages
        ))

    return blocks
