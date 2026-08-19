"""
Lightweight web UI for testing the RAG chat (ticket #17).

Wraps the EXISTING pipeline — ingest_and_populate_store + Retriever + Generator —
with a thin Flask frontend. No new pipeline logic: this mirrors scripts/chat.py
exactly, one retrieval + generation per question, no cross-turn memory. The
chat history you see is a visual log rendered in the browser (ticket #18 adds
localStorage persistence on the client side).

Same env-var configuration as the CLI:
    CHROMA_HOST      — vector store host (used via config.yaml)
    MODEL_ENDPOINT   — Ollama/Gemma host
    MODEL_TAG        — override the configured model tag

Run from repo root:
    pip install -e ".[webui]"
    python -m heinzy.webui.app
    # then open http://localhost:5000

The heavy objects (store, retriever, generator) are built ONCE at startup, the
same as chat.py builds them once before its loop. Each HTTP request only does
retrieve + generate.
"""
from __future__ import annotations

from dataclasses import dataclass

from flask import Flask, jsonify, render_template, request

# NOTE: heavy pipeline imports (ingest chain, generator) are deliberately done
# INSIDE build_engine(), not at module top. Importing this module must stay
# cheap so the app can be imported for testing/health without a reachable
# store or model, and so the lazy-build path actually defers that cost.

# Cues that indicate the model declined rather than answered. Kept in sync with
# Retained for diagnostics only. "declined" now comes from Answer.refused, which
# the generator sets when a refusal layer actually fired, so the UI no longer
# guesses at the model's wording.
_ABSTAIN_CUES = [
    "cannot", "can't", "not covered", "no information", "not in the",
    "does not contain", "doesn't contain", "unable to", "not provided",
    "not found", "not available", "outside", "no relevant",
]


def _looks_declined(text: str) -> bool:
    """Keyword guess at whether an answer was a refusal.

    No longer decides anything. The API reports Answer.refused, which the
    generator sets when a refusal layer actually fired, so the UI reads a fact
    instead of inferring one from prose. Kept because it is still useful for
    spotting disagreement between what the model wrote and what the system
    decided, and because a confident fabrication trips none of these cues while
    a legitimate answer containing "outside" trips one.
    """
    t = text.lower()
    return any(c in t for c in _ABSTAIN_CUES)


@dataclass
class _Engine:
    """Holds the built-once pipeline objects, mirroring chat.py's setup."""
    retriever: "object"   # heinzy.retrieval.retrieve.Retriever (imported lazily)
    generator: "object"   # heinzy.generation.generator.Generator (imported lazily)
    model_tag: str
    chunk_count: int


def build_engine() -> _Engine:
    from heinzy.common.config import load_config
    from heinzy.generation.generator import Generator
    from heinzy.pipeline import ingest_and_populate_store
    from heinzy.retrieval.retrieve import Retriever

    cfg = load_config()
    store = ingest_and_populate_store(cfg)
    retriever = Retriever(cfg, store=store)
    generator = Generator(cfg)
    return _Engine(
        retriever=retriever,
        generator=generator,
        model_tag=generator.model_tag,
        chunk_count=store.count(),
    )


def create_app(engine: _Engine | None = None) -> Flask:
    app = Flask(__name__)
    # Lazy-build so importing the module (e.g. for tests) doesn't require a
    # reachable store/model. Built on first request if not injected.
    app.config["_engine"] = engine

    def _get_engine() -> _Engine:
        if app.config["_engine"] is None:
            app.config["_engine"] = build_engine()
        return app.config["_engine"]

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        eng = app.config["_engine"]
        ready = eng is not None
        return jsonify({
            "ready": ready,
            "model_tag": eng.model_tag if ready else None,
            "chunks_indexed": eng.chunk_count if ready else None,
        })

    @app.route("/api/ask", methods=["POST"])
    def ask():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question must be non-empty"}), 400

        eng = _get_engine()
        result = eng.retriever.retrieve(question)
        answer = eng.generator.generate(result.query, result.hits)

        sources = [
            {
                "section_path": h.section_path,
                "source_pages": h.source_pages,
                "score": round(h.score, 4),
            }
            for h in answer.sources
        ]
        return jsonify({
            "question": question,
            "answer": answer.text,
            # Structured, not inferred. Set by whichever refusal layer fired.
            "declined": answer.refused,
            "decline_reason": answer.refusal_reason,
            "model_tag": answer.model_tag,
            "sources": sources,
        })

    return app


def main() -> None:
    import os

    # Build eagerly at startup so the first user request is fast and any
    # store/model misconfig surfaces immediately in the console, like chat.py.
    engine = build_engine()
    print(f"ready — {engine.chunk_count} chunks indexed, model={engine.model_tag}")
    app = create_app(engine)
    app.run(
        host=os.environ.get("HEINZY_UI_HOST", "0.0.0.0"),
        port=int(os.environ.get("HEINZY_UI_PORT", "5000")),
        debug=False,
    )


if __name__ == "__main__":
    main()
