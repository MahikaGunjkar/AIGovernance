"""
Full RAG loop: ingest  handbook, retrieve relevant chunks, generate a
grounded answer via a local Gemma model over Ollama. Builds on
ingest_and_retrieve_demo.py (ingest + retrieval) by adding generation.

Primary workflow: an advisor asking a question (often on a student's behalf),
matching the advisor/student/system Actor roles used elsewhere (A5 eventlog).

Run from repo root (needs Ollama running locally):
    python scripts/ask_handbook.py --query "how many electives can I take?"
    # to use a locally-pulled model that differs from config.yaml's model.tag:
    MODEL_TAG=gemma12:latest python scripts/ask_handbook.py --query "..."
"""
from __future__ import annotations

import argparse

from heinzy.common.config import load_config
from heinzy.generation.generator import Generator
from heinzy.pipeline import ingest_and_populate_store
from heinzy.retrieval.retrieve import Retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="What are the required courses?")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    store = ingest_and_populate_store(cfg)
    print(f"\nstore populated: {store.count()} chunks\n")

    retriever = Retriever(cfg, store=store)
    result = retriever.retrieve(args.query, k=args.k)

    generator = Generator(cfg)
    answer = generator.generate(result.query, result.hits)

    print(f"model    : {answer.model_tag}")
    print(f"question : {answer.query}\n")
    print(f"answer:\n{answer.text}\n")
    print("sources:")
    for h in answer.sources:
        print(f"  - [{h.section_path}] p{h.source_pages}  (score={h.score:.4f})")


if __name__ == "__main__":
    main()
