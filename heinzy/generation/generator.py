"""
IMPORTANT::!!
Answer generation over retrieved chunks.

pre: hits is a list of ScoredChunk from a completed retrieval (may be empty --
     the model is instructed to say so rather than guess)
post: returns an Answer carrying the model's text plus the exact ScoredChunks
      it was grounded in, so the caller can still cite/verify sources
invariant: model_tag and endpoint are read from config, never hardcoded --
           mirrors the "no tunable constants in source" rule. A local MODEL_TAG
           env var overrides config.model.tag for testing against whatever is
           actually pulled on this machine, without touching the shared config.

Talks to Ollama's /api/chat endpoint. endpoint comes from config.model.endpoint
(itself sourced from MODEL_ENDPOINT in .env); if that resolves empty, falls
back to the local Ollama default so this runs with zero setup on a dev machine
that already has Ollama running.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from heinzy.retrieval.store import ScoredChunk

_DEFAULT_ENDPOINT = "http://localhost:11434"

# Primary workflow is an advisor looking up policy on a student's behalf (see
# the advisor/student/system Actor roles in heinzy/eventlog/actor.py and the
# --actor-role default in scripts/smoke_retrieval.py) -- not a student asking
# the assistant directly. Keep the framing accurate to that audience.
SYSTEM_PROMPT = (
    "You are an assistant supporting academic advisors for the Heinz College "
    "MISM program. An advisor is asking you a question, often on behalf of a "
    "student, and needs an accurate answer they can rely on when advising. "
    "Answer using ONLY the provided handbook excerpts below. If the excerpts "
    "don't contain the answer, say so plainly -- do not guess or use outside "
    'knowledge. Cite the section(s) you used, e.g. (see "4.1. Required Courses"). '
    "Base your answer on ALL relevant information present in the excerpts -- do "
    "not silently omit or summarize down a subset of what's there. An advisor "
    "relying on an incomplete answer could give a student incorrect information."
)


@dataclass
class Answer:
    query: str
    text: str
    model_tag: str
    sources: list[ScoredChunk]


class Generator:
    def __init__(self, cfg) -> None:
        self.model_tag = os.environ.get("MODEL_TAG") or cfg.model.tag
        self.endpoint = (getattr(cfg.model, "endpoint", "") or _DEFAULT_ENDPOINT).rstrip("/")

    def generate(self, query: str, hits: list[ScoredChunk]) -> Answer:
        context = "\n\n".join(
            f'[{h.section_path}] (p{h.source_pages}): {h.text}' for h in hits
        )
        user_prompt = f"Handbook excerpts:\n\n{context}\n\nQuestion: {query}"

        resp = requests.post(
            f"{self.endpoint}/api/chat",
            json={
                "model": self.model_tag,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        return Answer(
            query=query,
            text=data["message"]["content"],
            model_tag=self.model_tag,
            sources=hits,
        )
