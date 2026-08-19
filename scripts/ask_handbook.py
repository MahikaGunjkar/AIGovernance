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
    ap.add_argument("--backend", default=None, choices=["memory", "chroma"],
                    help="override vector_store.backend for this run only")
    args = ap.parse_args()

    cfg = load_config()
    if args.backend is not None:
        cfg.vector_store.backend = args.backend
    store = ingest_and_populate_store(cfg)
    print(f"\nstore populated: {store.count()} chunks\n")

    retriever = Retriever(cfg, store=store)
    result = retriever.retrieve(args.query, k=args.k)

    generator = Generator(cfg)
    answer = generator.generate(result.query, result.hits)

    print(f"model    : {answer.model_tag}")
    print(f"question : {answer.query}\n")
    if answer.paused_for_approval:
        print("status   : PAUSED_FOR_APPROVAL (governance HITL)\n")
    print(f"answer:\n{answer.text}\n")
    print("sources:")
    print(f"question : {answer.query}")

    print(f"\nanswer:\n{answer.text}\n")
    if answer.refused:
        print(f"(no answer given, reason {answer.refusal_reason})")
        if answer.sources:
            print("closest sections (did not answer the question):")
    else:
        if answer.unsupported_citations:
            print("WARNING: answer cited sections that retrieval did not return: "
                  f"{', '.join(answer.unsupported_citations)}")
        print("sources:")
    for h in answer.sources:
        print(f"  - [{h.section_path}] p{h.source_pages}  (score={h.score:.4f})")
    if answer.tool_events:
        print("\ntool events:")
        for ev in answer.tool_events:
            print(f"  - {ev.get('status')}: {ev.get('tool')} {ev.get('args')}")


if __name__ == "__main__":
    main()
