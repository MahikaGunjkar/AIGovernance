"""
Shared abstention/decline detector.

Single source of truth for "did the model decline to answer?" so the eval
harness (eval/run_eval.py) and the web UI (heinzy/webui/app.py) can never drift
apart. Matches negation + a reporting verb ("does not say", "do not contain
information") plus a set of standalone cue phrases. Verb-based matching is
resilient to the phrasing varying run to run.

NOTE: this is a text heuristic on model output, not a deterministic policy gate.
See the refusal-policy-engine bug ticket — the durable fix is to key
abstention off retrieval signal and return a structured flag, with this
detector kept only as a display fallback.
"""
from __future__ import annotations

import re

_NEG = (
    r"(?:does not|doesn't|do not|don't|cannot|can't|could not|couldn't|"
    r"is not|isn't|are not|aren't|no|not)"
)
_ABSTAIN_VERB = (
    r"(?:say|state|mention|contain|include|provide|specify|indicate|"
    r"discuss|address|cover|list|describe|detail)"
)

_ABSTAIN_PATTERNS = [
    re.compile(rf"{_NEG}\s+\w*\s*{_ABSTAIN_VERB}"),
    re.compile(rf"{_ABSTAIN_VERB}\w*\s+{_NEG}\b"),
]

_ABSTAIN_CUES = [
    "cannot", "can't", "not covered", "no information", "not in the",
    "unable to", "not provided", "not found", "not available",
    "no relevant", "outside the scope", "insufficient information",
    "i don't have", "i do not have",
]


def looks_like_abstention(text: str) -> bool:
    """True if the answer text reads as a refusal/abstention rather than a
    substantive answer."""
    t = (text or "").lower()
    if any(c in t for c in _ABSTAIN_CUES):
        return True
    return any(p.search(t) for p in _ABSTAIN_PATTERNS)
