"""
Run a fixed set of edge-case questions through the full RAG pipeline (ingest
if needed -> retrieve -> generate), printing each result as it runs and
logging everything to a CSV for later review. Output is gitignored
(data/logs/*) -- this is a QA artifact, not committed data.

Run from repo root:
    CHROMA_HOST=127.0.0.1 MODEL_TAG=gemma12:latest python scripts/test_edge_cases.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from heinzy.common.config import load_config
from heinzy.generation.generator import Generator
from heinzy.pipeline import ingest_and_populate_store
from heinzy.retrieval.retrieve import Retriever

OUT_PATH = Path("data/logs/edge_case_results.csv")

EDGE_CASES = [
    "What is the tuition cost for the MISM program?",
    "What's the difference between the 12-month and 16-month MISM tracks?",
    "wat is the interships requiremnt if i alredy hav wrk experiance",
    "Who is the president of Carnegie Mellon University?",
    "What are the required core courses and their units for the MISM 12-month track?",
]


def main() -> None:
    cfg = load_config()
    store = ingest_and_populate_store(cfg)
    print(f"store populated: {store.count()} chunks\n")

    retriever = Retriever(cfg, store=store)
    generator = Generator(cfg)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "top_sections", "top_scores", "model_tag"])

        for i, question in enumerate(EDGE_CASES, 1):
            print(f"=== {i}/{len(EDGE_CASES)}: {question} ===")
            result = retriever.retrieve(question)
            answer = generator.generate(result.query, result.hits)

            sections = "; ".join(h.section_path or "unresolved" for h in answer.sources)
            scores = "; ".join(f"{h.score:.4f}" for h in answer.sources)

            print(answer.text)
            print()

            writer.writerow([question, answer.text, sections, scores, answer.model_tag])

    print(f"logged to {OUT_PATH}")


if __name__ == "__main__":
    main()
