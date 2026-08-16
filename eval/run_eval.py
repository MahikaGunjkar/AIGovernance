"""
Evaluation harness (Prototype task A6).

Runs the RAG pipeline over a curated question set and scores two things:

  1. Retrieval quality
       - hit@k : did a chunk from the expected section make the top-k?
       - MRR   : 1/rank of the first correct-section hit (0 if none)
  2. Answer correctness (only if generation is on)
       - graded WITHOUT an external API key. Default grader is embedding
         cosine similarity between the model answer and the ground-truth
         answer, using the SAME local Embedder the retriever uses (zero new
         deps, instant). `--judge ollama` optionally asks the shared Gemma
         host for a 0-1 score instead.
   Out-of-corpus questions (in_corpus=false) are scored on abstention: the
   system passes if it declines to answer rather than fabricating.

"done when" (A6): runs from one command, writes a result file tagged with the
config hash and the model build.

Run from repo root:
    # retrieval only, fast, no Gemma needed:
    python eval/run_eval.py --no-generation

    # full: retrieval + generation + similarity grading (needs Gemma reachable):
    python eval/run_eval.py

    # use the LLM as judge instead of embedding similarity:
    python eval/run_eval.py --judge ollama

    # custom set / threshold:
    python eval/run_eval.py --questions eval/questions.jsonl --sim-threshold 0.6
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from heinzy.common.config import load_config
from heinzy.retrieval.embedder import Embedder
from heinzy.retrieval.retrieve import Retriever

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")


# --------------------------------------------------------------------------- #
# grading helpers
# --------------------------------------------------------------------------- #
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _looks_like_abstention(text: str) -> bool:
    """Heuristic: did the model decline rather than fabricate?"""
    t = text.lower()
    cues = [
        "cannot", "can't", "not covered", "no information", "not in the",
        "does not contain", "doesn't contain", "unable to", "not provided",
        "not found", "not available", "outside", "no relevant",
    ]
    return any(c in t for c in cues)


def _section_hit(hits, expected_section_contains: str | None) -> tuple[bool, int]:
    """Return (hit, rank1) — did a hit's section_path contain the expected
    marker, and at what 1-based rank (0 if no hit)."""
    if not expected_section_contains:
        return (False, 0)
    needle = expected_section_contains.lower()
    for i, h in enumerate(hits, start=1):
        section = (h.section_path or "").lower()
        if needle in section:
            return (True, i)
    return (False, 0)


def _judge_ollama(cfg, question: str, expected: str, got: str) -> float:
    """Optional LLM judge via the shared Ollama host. Returns 0-1."""
    import os

    import requests

    endpoint = (getattr(cfg.model, "endpoint", "") or "http://localhost:11434").rstrip("/")
    model_tag = os.environ.get("MODEL_TAG") or cfg.model.tag
    prompt = (
        "You are grading an answer for factual agreement with a reference.\n"
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Given answer: {got}\n\n"
        "Reply with ONLY a number from 0.0 to 1.0 where 1.0 means the given "
        "answer is fully correct and complete versus the reference, and 0.0 "
        "means wrong or unrelated. No words, just the number."
    )
    resp = requests.post(
        f"{endpoint}/api/chat",
        json={
            "model": model_tag,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()
    # pull the first float out of the reply
    for token in raw.replace(",", " ").split():
        try:
            val = float(token)
            return max(0.0, min(1.0, val))
        except ValueError:
            continue
    return 0.0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_questions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--k", type=int, default=None, help="override retrieval k")
    ap.add_argument("--no-generation", action="store_true",
                    help="score retrieval only; skip Gemma (fast, no host needed)")
    ap.add_argument("--judge", choices=["similarity", "ollama"], default="similarity",
                    help="answer-correctness grader (default: embedding similarity)")
    ap.add_argument("--sim-threshold", type=float, default=0.55,
                    help="cosine >= this counts as a correct answer")
    args = ap.parse_args()

    cfg = load_config()
    questions = load_questions(args.questions)

    # Build the store once (real ingest) unless we're memory-mode/offline.
    # Import here so --no-generation retrieval-only runs don't require Ollama libs.
    from heinzy.pipeline import ingest_and_populate_store

    store = ingest_and_populate_store(cfg)
    retriever = Retriever(cfg, store=store)
    grader_embedder = Embedder(cfg.embed.model_tag, cfg.embed.dimension)

    generator = None
    if not args.no_generation:
        from heinzy.generation.generator import Generator
        generator = Generator(cfg)

    model_build = "none (retrieval-only)" if args.no_generation else (
        (generator.model_tag if generator else cfg.model.tag)
    )

    per_q = []
    t0 = time.time()
    for row in questions:
        qid = row["id"]
        question = row["question"]
        in_corpus = row.get("in_corpus", True)
        expected_answer = row.get("expected_answer", "")
        expected_section = row.get("expected_section_contains")

        result = retriever.retrieve(question, k=args.k)
        hit, rank1 = _section_hit(result.hits, expected_section)
        rr = (1.0 / rank1) if rank1 > 0 else 0.0

        rec = {
            "id": qid,
            "question": question,
            "in_corpus": in_corpus,
            "k": result.k,
            "retrieval_hit": hit if in_corpus else None,
            "reciprocal_rank": rr if in_corpus else None,
            "top_sections": [h.section_path for h in result.hits],
            "top_scores": [round(h.score, 4) for h in result.hits],
        }

        if generator is not None:
            answer = generator.generate(question, result.hits)
            got = answer.text
            rec["answer"] = got

            if in_corpus:
                if args.judge == "ollama":
                    score = _judge_ollama(cfg, question, expected_answer, got)
                else:
                    ev = grader_embedder.embed(expected_answer)
                    gv = grader_embedder.embed(got)
                    score = _cosine(ev, gv)
                rec["answer_score"] = round(score, 4)
                rec["answer_correct"] = bool(score >= args.sim_threshold)
            else:
                # out-of-corpus: correct == abstained
                abstained = _looks_like_abstention(got)
                rec["abstained"] = abstained
                rec["answer_correct"] = abstained

        per_q.append(rec)

    elapsed = round(time.time() - t0, 2)

    # -------- aggregate --------
    in_corpus_q = [r for r in per_q if r["in_corpus"]]
    out_q = [r for r in per_q if not r["in_corpus"]]

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "n_questions": len(per_q),
        "n_in_corpus": len(in_corpus_q),
        "n_out_of_corpus": len(out_q),
        "hit_at_k": _mean([r["retrieval_hit"] for r in in_corpus_q]),
        "mrr": _mean([r["reciprocal_rank"] for r in in_corpus_q]),
        "elapsed_seconds": elapsed,
    }
    if generator is not None:
        summary["answer_correctness"] = _mean(
            [r.get("answer_correct") for r in in_corpus_q]
        )
        if out_q:
            summary["abstention_rate"] = _mean(
                [r.get("abstained") for r in out_q]
            )
        summary["grader"] = args.judge
        summary["sim_threshold"] = args.sim_threshold

    # -------- write tagged result file (A6 "done when") --------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"eval_{cfg.config_hash}_{stamp}.json"
    payload = {
        "config_hash": cfg.config_hash,
        "config_version": cfg.raw.get("version"),
        "model_build": model_build,
        "embed_model": cfg.embed.model_tag,
        "generation": not args.no_generation,
        "timestamp_utc": stamp,
        "summary": summary,
        "results": per_q,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    # -------- console report --------
    print(f"\n=== Heinzy eval (A6) ===")
    print(f"config_hash : {cfg.config_hash}   model_build: {model_build}")
    print(f"questions   : {summary['n_questions']} "
          f"({summary['n_in_corpus']} in-corpus, {summary['n_out_of_corpus']} out)")
    print(f"hit@k       : {summary['hit_at_k']}")
    print(f"MRR         : {summary['mrr']}")
    if generator is not None:
        print(f"answer corr.: {summary.get('answer_correctness')} "
              f"(grader={args.judge}, thr={args.sim_threshold})")
        if out_q:
            print(f"abstention  : {summary.get('abstention_rate')}")
    print(f"elapsed     : {elapsed}s")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
