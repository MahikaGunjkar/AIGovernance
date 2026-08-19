"""
Prove the grounded-answering claim. On questions the corpus cannot
answer, the system says it cannot answer instead of producing plausible text.

Exits non-zero if any out-of-corpus question got answered, any in-corpus
control got refused, or any answer cited a section retrieval never returned,
so this can gate a PR.

Run from repo root:
    # against the real handbook in data/corpus/, via the shared Chroma host
    python scripts/eval_abstention.py

    # same, but no Chroma reachable, so ingest into an in-process store
    python scripts/eval_abstention.py --backend memory

    # with no handbook PDF present, using the labelled stand-in corpus
    python scripts/eval_abstention.py --fixture

    # sweep the retrieval gate without editing config.yaml
    python scripts/eval_abstention.py --fixture --score-floor 0.55

    # whatever model is actually pulled locally
    MODEL_TAG=llama3.2:latest python scripts/eval_abstention.py --fixture
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from heinzy.common.config import load_config
from heinzy.eval.abstention import (
    QUESTIONS_PATH,
    build_fixture_store,
    load_questions,
    run_eval,
    summary_lines,
    write_csv,
)

DEFAULT_OUT = Path("data/logs/abstention_eval.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, choices=["memory", "chroma"],
                    help="override vector_store.backend for this run only "
                         "(e.g. memory, when the shared Chroma host is unreachable)")
    ap.add_argument("--fixture", action="store_true",
                    help="use the labelled stand-in corpus instead of data/corpus/")
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--score-floor", type=float, default=None,
                    help="override retrieval.score_floor for this run only")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    cfg = load_config()
    if args.backend is not None:
        cfg.vector_store.backend = args.backend
    if args.score_floor is not None:
        # Experiment override. Recorded in the report and CSV, because with it
        # set the config_hash alone no longer describes the run.
        cfg.retrieval.score_floor = args.score_floor

    if args.fixture:
        print("=" * 72)
        print("FIXTURE MODE. Stand-in corpus, NOT the real MISM handbook.")
        print("Proves the refusal machinery works. Says nothing about the real")
        print("handbook's coverage. Do not report these numbers as handbook results.")
        print("=" * 72)
        store = build_fixture_store(cfg)
    else:
        # Imported here, not at module scope: fixture mode exists for machines
        # without the ingest stack (CI, fresh clone), and pulling in pymupdf +
        # tiktoken at import time would defeat that.
        from heinzy.pipeline import ingest_and_populate_store

        store = ingest_and_populate_store(cfg)
    print(f"store: {store.count()} chunks\n")

    questions = load_questions(args.questions)

    def show(r) -> None:
        mark = "PASS" if r.passed else "FAIL"
        verdict = "refused" if r.refused else "answered"
        score = "  n/a" if r.top_score is None else f"{r.top_score:.3f}"
        print(f"[{mark}] {r.question.id:<7} {verdict:<8} top={score}  {r.question.question[:64]}")
        if not r.passed:
            print(f"       -> {r.failure}")
            print(f"       -> got: {r.answer_text[:200].strip()}")

    print(f"{'':7} {'id':<7} {'outcome':<8} {'top':<10} question")
    report = run_eval(
        cfg, store, questions, k=args.k, fixture_mode=args.fixture, on_result=show
    )

    print()
    for line in summary_lines(report):
        print(line)

    write_csv(report, args.out)
    print(f"\nwrote {args.out}")
    print("RESULT: PASS" if report.passed else "RESULT: FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
