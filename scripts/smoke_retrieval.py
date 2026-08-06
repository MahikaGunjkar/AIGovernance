"""
Smoke test for the retrieval layer (A2).

Seeds a handful of placeholder policy chunks into the in-memory store, then runs
a query and prints ranked hits with provenance. Proves the pipeline end to end
WITHOUT the real handbook or a DB. Swap in real ingest output + a real store
later; this script does not change.

Run from repo root:
    python scripts/smoke_retrieval.py
    python scripts/smoke_retrieval.py --k 2 --query "how many electives"
"""
from __future__ import annotations

import argparse
import json
import os

from heinzy.common.config import load_config
from heinzy.eventlog import Actor, get_event_log
from heinzy.retrieval.embedder import Embedder
from heinzy.retrieval.retrieve import Retriever
from heinzy.retrieval.store import InMemoryStore, StoredChunk

# Placeholder stand-ins for real MISM handbook chunks. Content is invented and
# clearly labeled so nobody mistakes it for real policy.
PLACEHOLDER = [
    ("c1", "MISM students must complete 144 units to graduate.", "MISM Handbook > Graduation", [12]),
    ("c2", "The core curriculum includes statistics, economics, and databases.", "MISM Handbook > Core", [8]),
    ("c3", "Students may take up to 4 elective courses outside Heinz.", "MISM Handbook > Electives", [21]),
    ("c4", "The internship requirement can be waived with prior work experience.", "MISM Handbook > Internship", [30]),
    ("c5", "Academic integrity violations are reported to the dean's office.", "MISM Handbook > Conduct", [45]),
]


def build_seeded_store(cfg) -> InMemoryStore:
    embedder = Embedder(cfg.embed.model_tag, cfg.embed.dimension)
    store = InMemoryStore()
    stored = [
        StoredChunk(
            chunk_id=cid,
            vector=embedder.embed(text),
            text=text,
            doc_id="placeholder-mism-handbook",
            section_path=section,
            source_pages=pages,
        )
        for cid, text, section, pages in PLACEHOLDER
    ]
    store.add(stored)
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="How many elective courses can I take?")
    ap.add_argument("--k", type=int, default=None, help="override config k")
    ap.add_argument(
        "--actor-id",
        default=os.environ.get("HEINZY_ACTOR_ID", "smoke-operator"),
        help="A5 actor_id stamped into the audit record",
    )
    ap.add_argument(
        "--actor-role",
        default=os.environ.get("HEINZY_ACTOR_ROLE", "advisor"),
        help="A5 actor role (advisor|student|system|...)",
    )
    args = ap.parse_args()

    cfg = load_config()
    store = build_seeded_store(cfg)
    event_log = get_event_log(cfg)
    actor = Actor(actor_id=args.actor_id, role=args.actor_role)
    retriever = Retriever(
        cfg, store=store, event_log=event_log, default_actor=actor
    )
    result = retriever.retrieve(args.query, k=args.k)

    print(f"config_hash : {cfg.config_hash}")
    print(f"embedder    : {result.embed_model} "
          f"({'semantic' if result.is_semantic else 'HASH-FALLBACK, not semantic'})")
    print(f"actor       : {actor.actor_id} ({actor.role})")
    print(f"k (effective): {result.k}   store size: {store.count()}")
    print(f"query       : {result.query}\n")
    for rank, h in enumerate(result.hits, 1):
        print(f"  {rank}. score={h.score:.4f}  [{h.section_path}] p{h.source_pages}")
        print(f"     {h.text}")
    print("\n--- audit record (task A5) ---")
    print(json.dumps(result.audit_record or result.to_log_record(), indent=2))
    if event_log is not None:
        print(f"\nappended to: {event_log.path.resolve()}")


if __name__ == "__main__":
    main()
