"""
Layer 2 of abstention, the insufficient-context check (A3b, split out per #11).

Layer 1 asks a structural question that a policy engine can answer on its own.
Did retrieval return enough to work with. Layer 2 asks a question no engine can
answer, which is whether the chunks that came back actually contain the answer.
Only the model can judge that, so it is instructed to emit an exact sentinel and
this module decides whether it did.

Keeping the two apart matters because they fail differently. Layer 1 is a
deterministic gate that can be audited and replayed. Layer 2 is a model
judgement that varies with the model and the prompt, and has to be measured per
model rather than assumed. Filing both under one `if` in the generator hid that
distinction.

pre:  raw is the model's unmodified reply, sentinel is the configured token
post: detects_refusal is True when the model signalled it could not answer
invariant: pure functions, no config and no I/O, so tests and the eval harness
           can call them without a model or a network

Note on what this is NOT. This does not check topical relevance. Relevance is
what retrieval.score_floor screens on in Layer 1. On the MISM handbook the
in-corpus and out-of-corpus questions overlap in similarity to within 0.0249,
so relevance cannot separate them and a chunk can score well while answering
nothing. That gap is exactly what Layer 2 exists to catch.
"""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def sentinel_key(text: str) -> str:
    """Uppercase, alphanumerics only, so spacing and markup cannot hide it."""
    return _NON_ALNUM.sub("", (text or "").upper())


def detects_refusal(raw: str, sentinel: str) -> bool:
    """True when the model signalled that the excerpts do not answer the question.

    Matched on the punctuation-stripped form rather than by equality. Told to
    reply with exactly INSUFFICIENT_CONTEXT, llama3.2 replies "INSUFFICIENT
    CONTEXT", gemma bolds it, and others wrap it in a sentence. An exact match
    scored every one of those as a real answer, which is the same as having no
    Layer 2 at all.
    """
    key = sentinel_key(sentinel)
    return bool(key) and key in sentinel_key(raw)
