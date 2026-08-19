
from __future__ import annotations

import argparse

from heinzy.common.config import load_config
from heinzy.generation.generator import Generator
from heinzy.pipeline import ingest_and_populate_store
from heinzy.retrieval.retrieve import Retriever

QUIT_WORDS = {"quit", "exit", "q"}


def print_answer(answer) -> None:
    """Render an answer, distinguishing a refusal from a grounded answer.

    On a refusal the retrieved chunks are still shown, but labelled as what
    they are, the closest sections, which did NOT answer the question.
    Printing them under "sources:" implies the refusal was drawn from them.
    """
    print(f"\nheinzy> {answer.text}\n")

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
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, choices=["memory", "chroma"],
                    help="override vector_store.backend for this run only")
    args = ap.parse_args()

    cfg = load_config()
    if args.backend is not None:
        cfg.vector_store.backend = args.backend
    store = ingest_and_populate_store(cfg)
    retriever = Retriever(cfg, store=store)
    generator = Generator(cfg)

    print(f"\nready — {store.count()} chunks indexed, model={generator.model_tag}")
    print('type a question, or "quit" to leave\n')

    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question.lower() in QUIT_WORDS:
            break

        print("  searching handbook...", end="", flush=True)
        result = retriever.retrieve(question)
        print(f" found {len(result.hits)} relevant sections.")

        print("  generating answer...", flush=True)
        answer = generator.generate(result.query, result.hits)

        if answer.paused_for_approval:
            print(f"\nheinzy> [PAUSED_FOR_APPROVAL] {answer.text}\n")
        else:
            print(f"\nheinzy> {answer.text}\n")
        print("sources:")
        for h in answer.sources:
            print(f"  - [{h.section_path}] p{h.source_pages}  (score={h.score:.4f})")
        print()
        print_answer(answer)


if __name__ == "__main__":
    main()
