"""
pre: none — runs on whatever exists
post: result.passed == (len(result.failures) == 0)
invariant: read-only, never mutates chunks or pages.
"""
from __future__ import annotations

from .types import Chunk, PageRecord, VerifyConfig, VerifyReport

_SHORT_CHUNK_THRESHOLD = 20


def verify(chunks: list[Chunk], pages: list[PageRecord], cfg: VerifyConfig) -> VerifyReport:
    failures = []
    page_doc_ids = {p.doc_id for p in pages}

    for chunk in chunks:
        if not chunk.text.strip():
            failures.append(f"chunk {chunk.chunk_id} has empty text")

        if chunk.doc_id not in page_doc_ids:
            failures.append(f"chunk {chunk.chunk_id} doc_id {chunk.doc_id} not found in pages")

        if len(chunk.text) < _SHORT_CHUNK_THRESHOLD:
            failures.append(
                f"chunk {chunk.chunk_id} is suspiciously short "
                f"({len(chunk.text)} chars): {chunk.text!r}"
            )

    return VerifyReport(passed=len(failures) == 0, failures=failures)
