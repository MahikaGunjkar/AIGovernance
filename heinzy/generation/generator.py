"""
Grounded answer generation over retrieved chunks (A3 + A3b).

pre: hits is a list of ScoredChunk from a completed retrieval, already filtered
     by retrieval.score_floor. It may be empty, which is the primary refusal path
post: returns an Answer carrying the model's text plus the exact ScoredChunks it
      was grounded in, so the caller can cite/verify sources. `refused` says
      whether the system declined to answer, and `refusal_reason` says which
      layer declined.
invariant: model_tag, endpoint, and every abstention knob are read from config,
           never hardcoded, mirroring the "no tunable constants in source" rule.
           A local MODEL_TAG env var overrides config.model.tag for testing
           against whatever is actually pulled on this machine.

Grounded answering means two things, enforced separately:

  1. Answers are built only from retrieved chunks. The prompt forbids outside
     knowledge, and every section the answer cites is checked against what
     retrieval actually returned (heinzy/generation/grounding.py).

  2. When the corpus does not contain the answer, the system SAYS SO instead of
     producing plausible text. Two layers, because either alone leaks:

       Layer 1 covers the no context case. Fewer than
       generation.abstain.min_hits survived retrieval.score_floor, so the model
       is never called. This is deterministic. A question the corpus cannot
       support never reaches the model at all.

       Layer 2 covers insufficient context. Chunks cleared the floor but do not
       actually answer the question. Only the model can judge that, so it is
       instructed to emit an exact sentinel, which we detect and convert into
       the same refusal. Nearest-neighbour search always returns *something*,
       so without this layer a well-scoring but irrelevant chunk is exactly
       what a confident, wrong answer gets built from.

A refusal returns the configured refusal_text, not model prose, so downstream
(eval harness, event log, UI) can branch on `Answer.refused` instead of pattern
matching on English.

Talks to Ollama's /api/chat. endpoint comes from config.model.endpoint (itself
sourced from MODEL_ENDPOINT in .env); if that resolves empty, falls back to the
local Ollama default so this runs on a dev machine that already has Ollama up.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import requests

from heinzy.generation.grounding import extract_citations, unsupported_citations
from heinzy.retrieval.store import ScoredChunk

_DEFAULT_ENDPOINT = "http://localhost:11434"
_DEFAULT_SENTINEL = "INSUFFICIENT_CONTEXT"
_DEFAULT_REFUSAL = (
    "I can't answer that from the MISM handbook. The handbook sections I can "
    "search don't contain the information needed to answer this question. "
    "Please check with Heinz College advising staff directly."
)

# Why the refusal reasons are named, not booleans: an advisor seeing "no
# handbook section came close" needs a different follow-up than "the handbook
# covers this area but not this detail", and the eval harness reports on which
# layer fired.
REASON_NO_CONTEXT = "no_retrieved_context"
REASON_INSUFFICIENT = "model_insufficient_context"

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def _sentinel_key(text: str) -> str:
    """Uppercase, alphanumerics only, so spacing and markup cannot hide it."""
    return _NON_ALNUM.sub("", (text or "").upper())

# The primary workflow is an advisor looking up policy on a student's behalf,
# not a student asking the assistant directly. Keep the framing accurate to that
# audience. An advisor repeating an invented policy to a student is the exact
# failure this prompt exists to prevent.
SYSTEM_PROMPT_TEMPLATE = (
    "You are an assistant supporting academic advisors for the Heinz College "
    "MISM program. An advisor is asking you a question, often on behalf of a "
    "student, and needs an accurate answer they can rely on when advising.\n\n"
    "RULES:\n"
    "1. Answer using ONLY the handbook excerpts provided below. Never use "
    "outside knowledge, general knowledge about universities, or anything you "
    "know about Carnegie Mellon that is not in the excerpts.\n"
    "2. If the excerpts do not contain the information needed to answer the "
    "question, reply with exactly {sentinel} and nothing else. Do not "
    "apologise, do not explain, do not offer a partial guess, and do not "
    "suggest what the answer is likely to be. A wrong answer is far worse than "
    "no answer, because the advisor will repeat it to a student.\n"
    "3. This applies even when the question sounds like something a handbook "
    "would obviously cover, and even when the excerpts are about a related "
    "topic. Related is not the same as answering the question.\n"
    # The example here is a PLACEHOLDER on purpose. An earlier version used a
    # realistic sample sentence about elective limits, and llama3.2 answered a
    # real question by copying it verbatim, inventing a section that does not
    # exist in this handbook. Never put plausible-looking content in the
    # instructions, because the model cannot tell your example from its
    # evidence.
    "4. When the excerpts answer the question, and only then, write the "
    "answer itself as a sentence, then cite the section(s) you used in the form "
    '(see "<exact section heading copied from the excerpts>"). A bare citation '
    "with no sentence is not an answer; the advisor needs the substance, not a "
    "pointer. Never cite a section that does not appear in the excerpts below, "
    "and never reuse a section name from these instructions.\n"
    "5. Rule 4 never overrides rule 2. If the excerpts do not answer the "
    f"question, {{sentinel}} is the only correct response. Writing a fluent "
    "sentence is not the goal, being right is.\n"
    "6. Base your answer on ALL relevant information present in the excerpts "
    "-- do not silently omit part of what's there. An advisor relying on an "
    "incomplete answer could give a student incorrect information."
)


@dataclass
class Answer:
    query: str
    text: str
    model_tag: str
    sources: list[ScoredChunk]
    # --- grounding / abstention contract (A3b) ---
    refused: bool = False
    refusal_reason: str | None = None
    cited_sections: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    # Model's untouched output, kept when `text` was replaced by refusal_text so
    # a reviewer can see what the model actually said.
    raw_text: str | None = None

    @property
    def is_grounded(self) -> bool:
        """True when the answer cites nothing outside the retrieved set.

        A refusal is trivially grounded, since it makes no claims at all.
        """
        return not self.unsupported_citations


class Generator:
    def __init__(self, cfg) -> None:
        self.model_tag = os.environ.get("MODEL_TAG") or cfg.model.tag
        self.endpoint = (getattr(cfg.model, "endpoint", "") or _DEFAULT_ENDPOINT).rstrip("/")

        # Abstention knobs live in config (S5). getattr chains keep this working
        # against an older config.yaml that predates the generation section.
        generation = getattr(cfg, "generation", None)
        abstain = getattr(generation, "abstain", None)
        self.sentinel = (getattr(abstain, "sentinel", None) or _DEFAULT_SENTINEL).strip()
        self.min_hits = int(getattr(abstain, "min_hits", 1) or 1)
        self.refusal_text = " ".join(
            (getattr(abstain, "refusal_text", None) or _DEFAULT_REFUSAL).split()
        )
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(sentinel=self.sentinel)
        self._sentinel_key = _sentinel_key(self.sentinel)

        citations = getattr(generation, "citations", None)
        self.require_citations = bool(getattr(citations, "require", True))

        self.temperature = float(getattr(generation, "temperature", 0.0) or 0.0)
        self.seed = getattr(generation, "seed", 0)

    def generate(self, query: str, hits: list[ScoredChunk]) -> Answer:
        # Layer 1: nothing relevant retrieved -> refuse without calling the
        # model. No context means no grounded answer is possible, so asking the
        # model at all only invites one that isn't.
        if len(hits) < self.min_hits:
            return self._refusal(query, hits, REASON_NO_CONTEXT)

        context = "\n\n".join(
            f'[{h.section_path}] (p{h.source_pages}): {h.text}' for h in hits
        )
        user_prompt = f"Handbook excerpts:\n\n{context}\n\nQuestion: {query}"

        resp = requests.post(
            f"{self.endpoint}/api/chat",
            json={
                "model": self.model_tag,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": self.temperature, "seed": self.seed},
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]

        # Layer 2: context cleared the floor but doesn't answer the question.
        # Matched on the punctuation-stripped form, not equality: told to reply
        # with exactly INSUFFICIENT_CONTEXT, llama3.2 replies "INSUFFICIENT
        # CONTEXT" and gemma wraps it in a sentence or bolds it. An exact match
        # scored those as answers, which is the same as having no layer 2 at
        # all, a silent and total failure of the refusal path. Match loosely.
        if self._sentinel_key and self._sentinel_key in _sentinel_key(raw):
            return self._refusal(query, hits, REASON_INSUFFICIENT, raw_text=raw)

        cited = extract_citations(raw)
        unsupported = (
            unsupported_citations(
                cited, [h.section_path for h in hits], [h.text for h in hits]
            )
            if self.require_citations
            else []
        )
        return Answer(
            query=query,
            text=raw,
            model_tag=self.model_tag,
            sources=hits,
            cited_sections=cited,
            unsupported_citations=unsupported,
        )

    def _refusal(
        self,
        query: str,
        hits: list[ScoredChunk],
        reason: str,
        raw_text: str | None = None,
    ) -> Answer:
        return Answer(
            query=query,
            text=self.refusal_text,
            model_tag=self.model_tag,
            sources=list(hits),
            refused=True,
            refusal_reason=reason,
            raw_text=raw_text,
        )
