
from __future__ import annotations

from heinzy.common.config import load_config
from heinzy.generation.generator import Generator
from heinzy.pipeline import ingest_and_populate_store
from heinzy.retrieval.retrieve import Retriever

QUIT_WORDS = {"quit", "exit", "q"}


def main() -> None:
    cfg = load_config()
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

        print(f"\nheinzy> {answer.text}\n")
        print("sources:")
        for h in answer.sources:
            print(f"  - [{h.section_path}] p{h.source_pages}  (score={h.score:.4f})")
        print()


if __name__ == "__main__":
    main()
