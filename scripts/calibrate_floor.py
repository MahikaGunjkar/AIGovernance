"""
Pick retrieval.score_floor from data instead of guessing. Layer 1 depends on it.

Nearest-neighbour search always returns k chunks, so there is no "no match".
The floor is what turns "nothing scored well" into a refusal, and it only works
if in-corpus and out-of-corpus questions actually separate by score.

Do not eyeball a number. Embedding similarities are not calibrated
probabilities: bge-small packs all text into a narrow cone, so two unrelated
sentences routinely score 0.6+. A floor that looks strict (0.3) can gate
nothing at all. This script measures the real distribution on YOUR corpus and
reports whether a separating floor exists.

Retrieval only, no model calls, so it is fast and free to re-run after
re-chunking.

Run from repo root:
    python scripts/calibrate_floor.py               # real corpus
    python scripts/calibrate_floor.py --fixture     # stand-in corpus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from heinzy.common.config import load_config
from heinzy.eval.abstention import (
    EXPECT_ANSWER,
    EXPECT_REFUSE,
    QUESTIONS_PATH,
    build_fixture_store,
    load_questions,
)
from heinzy.retrieval.retrieve import Retriever


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, choices=["memory", "chroma"],
                    help="override vector_store.backend for this run only "
                         "(e.g. memory, when the shared Chroma host is unreachable)")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.backend is not None:
        cfg.vector_store.backend = args.backend
    # Measure the raw distribution: an existing floor would hide the scores
    # this script exists to show.
    cfg.retrieval.score_floor = 0.0

    if args.fixture:
        store = build_fixture_store(cfg)
    else:
        from heinzy.pipeline import ingest_and_populate_store  # heavy ingest deps

        store = ingest_and_populate_store(cfg)
    if args.fixture:
        print("FIXTURE MODE. Calibrate against the real handbook before trusting a floor.\n")

    retriever = Retriever(cfg, store=store)
    if not retriever.embedder.is_semantic:
        print("ABORT: embedder fell back to hash embedding. Scores are noise; "
              "install fastembed before calibrating.")
        return 2

    scored: dict[str, list[tuple[float, str]]] = {EXPECT_ANSWER: [], EXPECT_REFUSE: []}
    for q in load_questions(args.questions):
        hits = retriever.retrieve(q.question, k=args.k).hits
        top = hits[0].score if hits else 0.0
        scored[q.expected].append((top, q.id))

    for label, key in (("IN-CORPUS (should score high)", EXPECT_ANSWER),
                       ("OUT-OF-CORPUS (should score low)", EXPECT_REFUSE)):
        print(f"{label}")
        for score, qid in sorted(scored[key], reverse=True):
            print(f"  {score:.4f}  {qid}")
        print()

    ic = [s for s, _ in scored[EXPECT_ANSWER]]
    ooc = [s for s, _ in scored[EXPECT_REFUSE]]
    if not ic or not ooc:
        print("Need both question groups to calibrate.")
        return 2

    lowest_keep, highest_drop = min(ic), max(ooc)
    print(f"lowest in-corpus top score  : {lowest_keep:.4f}")
    print(f"highest out-of-corpus score : {highest_drop:.4f}")
    print()

    if lowest_keep > highest_drop:
        suggested = round((lowest_keep + highest_drop) / 2, 3)
        print(f"Separable. Suggested retrieval.score_floor: {suggested}")
        print(f"Margin is {lowest_keep - highest_drop:.4f}. Thin margins overfit "
              f"to these {len(ic) + len(ooc)} questions, so re-check after any "
              "change to chunking, k, or the embedding model. A thin margin is a "
              "reason to keep the floor loose, not to paste this number in.")
    else:
        print("NOT separable: at least one out-of-corpus question outscores a real "
              "one, so no floor gates the bad without gating the good.")
        print("Keep score_floor low and let layer 2 (the model sentinel) carry the "
              "refusals. That is the expected outcome for a single-document corpus, "
              "not a bug, since every question is 'about' the handbook's subject "
              "matter.")

    print("\nfloor      keeps in-corpus   gates out-of-corpus")
    for floor in [round(0.05 * i, 2) for i in range(0, 21)]:
        keeps = sum(1 for s in ic if s >= floor)
        gates = sum(1 for s in ooc if s < floor)
        if keeps == 0 and floor > lowest_keep:
            break
        print(f"{floor:<10.2f} {keeps}/{len(ic):<15} {gates}/{len(ooc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
