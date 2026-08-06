"""
Shared record types for the ingest pieline

Fields here are needed to exprss the pre/post contracts in
each module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SectionPath = str | Literal["unresolved"]


# ---------------------------------------------------------------- M0 --
@dataclass(frozen=True)
class DocEntry:
    doc_id: str  # == sha256, deterministic across machines
    sha256: str
    pdf_path: Path
    page_count: int


@dataclass
class Manifest:
    entries: dict[str, DocEntry] = field(default_factory=dict)  # keyed by sha256


# ---------------------------------------------------------------- M1 --
@dataclass
class ExtractConfig:
    ...  # TODO: extraction knobs (e.g. ocr_fallback: bool)


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    pdf_page_index: int
    text: str


# ---------------------------------------------------------------- M2 --
@dataclass
class StructureConfig:
    ...  # TODO: outline-vs-heading merge knobs


@dataclass(frozen=True)
class Block:
    doc_id: str
    text: str
    section_path: SectionPath | None
    source_pages: list[int]


# ---------------------------------------------------------------- M3 --
@dataclass
class ChunkConfig:
    target_tokens: int
    overlap_tokens: int = 0


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    section_path: SectionPath | None
    is_table: bool = False


# ---------------------------------------------------------------- M4 --
@dataclass
class EmbedConfig:
    model_tag: str
    dimension: int


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk_id: str
    vector: list[float]
    model_tag: str


# ---------------------------------------------------------------- M5 --
@dataclass
class IndexConfig:
    ...  # TODO: vector-store-specific knobs; hashed into collection name


@dataclass(frozen=True)
class Collection:
    name: str  # == hash of the full config chain
    config_hash: str


# ---------------------------------------------------------------- M6 --
@dataclass
class VerifyConfig:
    ...  # TODO: which checks to run, thresholds


@dataclass(frozen=True)
class VerifyReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
