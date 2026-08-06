"""
pre: none — runs on whatever exists
post: result.passed == (len(result.failures) == 0)
invariant: read-only, never mutates chunks or pages.
"""
from __future__ import annotations

from .types import Chunk, PageRecord, VerifyConfig, VerifyReport


def verify(chunks: list[Chunk], pages: list[PageRecord], cfg: VerifyConfig) -> VerifyReport:
    raise NotImplementedError
