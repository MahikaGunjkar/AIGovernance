"""
Grounded answer generation over retrieved chunks.

pre: hits is a list of ScoredChunk from a completed retrieval, already filtered
     by retrieval.score_floor. It may be empty, which is the primary refusal path.
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

       Layer 1 covers the no-context case. Fewer than
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

When governance is available (heinzy/tools mount present and not disabled in
config), generate() enters an Ollama tool-calling loop. Every tool_call is gated
by run_governed_tool before the tool runs.

Talks to Ollama's /api/chat. endpoint comes from config.model.endpoint (itself
sourced from MODEL_ENDPOINT in .env); if that resolves empty, falls back to the
local Ollama default so this runs on a dev machine that already has Ollama up.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from heinzy.generation.abstain import detects_refusal
from heinzy.generation.grounding import extract_citations, unsupported_citations
from heinzy.generation.policy import REASON_NO_CONTEXT, get_context_policy
from heinzy.governance.loader import governance_available
from heinzy.retrieval.store import ScoredChunk
from heinzy.tools.registry import TOOL_DEFINITIONS, run_governed_tool

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
# layer fired. REASON_NO_CONTEXT is re-exported from the policy module so
# callers keep one import site for both layers' reasons.
REASON_INSUFFICIENT = "model_insufficient_context"


def _basic_auth_from_env() -> tuple[str, str] | None:
    """Read MODEL_BASIC_AUTH as user:password, or None when the tunnel is open."""
    raw = (os.environ.get("MODEL_BASIC_AUTH") or "").strip()
    if not raw or ":" not in raw:
        return None
    user, _, password = raw.partition(":")
    return user, password


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
    "question, {sentinel} is the only correct response. Writing a fluent "
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
    paused_for_approval: bool = False
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    # --- grounding / abstention contract ---
    refused: bool = False
    refusal_reason: str | None = None
    cited_sections: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    # Model's untouched output, kept when `text` was replaced by refusal_text so
    # a reviewer can see what the model actually said.
    raw_text: str | None = None
    # Which Layer 1 engine decided, builtin or agt. Stamped so a run's audit
    # trail says whether the policy engine was actually in the loop.
    policy_engine: str | None = None

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

        # The shared host runs behind a tunnel, and Ollama has no auth of its
        # own, so the tunnel should carry basic auth. Credentials come from the
        # environment and never from config, which is committed.
        self.auth = _basic_auth_from_env()

        # Tool loop is enabled only when the governance mount is present and
        # config does not disable it.
        gov = getattr(cfg, "governance", None)
        self._governance_cfg = gov
        enabled = True if gov is None else bool(getattr(gov, "enabled", True))
        self.use_tools = bool(enabled and governance_available())
        self.max_tool_rounds = int(getattr(gov, "max_tool_rounds", 3) if gov else 3)
        self.agent_id = str(getattr(gov, "agent_id", "heinzy-advisor") if gov else "heinzy-advisor")

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
        self.system_prompt_with_tools = (
            self.system_prompt
            + "\n\nYou may call tools when needed. Write/create/update/delete/insert "
            "actions are forbidden. Web search is only for official CMU domains and "
            "may require human approval. Prefer handbook excerpts over tools when "
            "they suffice."
        )

        # Layer 1 lives in a policy object rather than an inline condition, so
        # the decision can be evaluated and audited by a governance engine.
        self.policy = get_context_policy(cfg)

        citations = getattr(generation, "citations", None)
        self.require_citations = bool(getattr(citations, "require", True))

        self.temperature = float(getattr(generation, "temperature", 0.0) or 0.0)
        self.seed = getattr(generation, "seed", 0)

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def _chat(self, messages: list[dict[str, Any]], *, tools: list[dict] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_tag,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        if tools:
            payload["tools"] = tools
        resp = requests.post(
            f"{self.endpoint}/api/chat",
            json=payload,
            timeout=180,
            auth=self.auth,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize Ollama tool_calls (name/arguments may be nested under 'function')."""
        raw = message.get("tool_calls") or []
        out: list[dict[str, Any]] = []
        for tc in raw:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name") or ""
            args = fn.get("arguments", tc.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"value": args}
            out.append({"name": name, "arguments": args, "raw": tc})
        return out

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, query: str, hits: list[ScoredChunk]) -> Answer:
        # Layer 1. Nothing relevant retrieved means no grounded answer is
        # possible, so the model is never called. Asking it anyway only invites
        # an answer that isn't grounded.
        decision = self.policy.decide(len(hits))
        if not decision.allowed:
            return self._refusal(
                query, hits, decision.reason, policy_engine=decision.engine
            )

        context = "\n\n".join(
            f"[{h.section_path}] (p{h.source_pages}): {h.text}" for h in hits
        )
        user_prompt = f"Handbook excerpts:\n\n{context}\n\nQuestion: {query}"

        if self.use_tools:
            raw, tool_events, paused = self._generate_with_tools(query, user_prompt, hits)
            if paused is not None:
                return paused
        else:
            data = self._chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
            )
            raw = data["message"]["content"]
            tool_events = []

        # Layer 2 lives in heinzy/generation/abstain.py. It answers a question
        # no policy engine can, which is whether the excerpts actually contain
        # the answer.
        if detects_refusal(raw, self.sentinel):
            return self._refusal(
                query, hits, REASON_INSUFFICIENT, raw_text=raw,
                policy_engine=decision.engine, tool_events=tool_events,
            )

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
            policy_engine=decision.engine,
            tool_events=tool_events,
        )

    def _generate_with_tools(
        self, query: str, user_prompt: str, hits: list[ScoredChunk]
    ) -> tuple[str, list[dict[str, Any]], Answer | None]:
        """Run the Ollama tool-calling loop.

        Returns (raw_text, tool_events, paused_answer). paused_answer is non-None
        only when a tool call paused for human approval, in which case the caller
        should return it directly and skip Layer 2.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt_with_tools},
            {"role": "user", "content": user_prompt},
        ]
        tool_events: list[dict[str, Any]] = []

        for _ in range(self.max_tool_rounds):
            data = self._chat(messages, tools=TOOL_DEFINITIONS)
            message = data.get("message") or {}
            messages.append(message)

            tool_calls = self._tool_calls_from_message(message)
            if not tool_calls:
                return message.get("content") or "", tool_events, None

            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]
                try:
                    result = run_governed_tool(
                        name, args, query=query, agent_id=self.agent_id
                    )
                except PermissionError as exc:
                    tool_events.append(
                        {"tool": name, "args": args, "status": "DENIED", "error": str(exc)}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps({"status": "DENIED", "error": str(exc)}),
                        }
                    )
                    continue

                if result.get("status") == "PAUSED_FOR_APPROVAL":
                    tool_events.append(
                        {"tool": name, "args": args, "status": "PAUSED_FOR_APPROVAL", "result": result}
                    )
                    paused = Answer(
                        query=query,
                        text=(
                            "Paused for human approval before completing a tool call. "
                            f"Details: {result.get('details')}"
                        ),
                        model_tag=self.model_tag,
                        sources=hits,
                        paused_for_approval=True,
                        tool_events=tool_events,
                    )
                    return "", tool_events, paused

                tool_events.append(
                    {"tool": name, "args": args, "status": "SUCCESS", "result": result}
                )
                messages.append(
                    {"role": "tool", "content": json.dumps(result)},
                )

        # Exhausted rounds — ask once more without tools for a final answer.
        data = self._chat(messages, tools=None)
        message = data.get("message") or {}
        return message.get("content") or "", tool_events, None

    def _refusal(
        self,
        query: str,
        hits: list[ScoredChunk],
        reason: str,
        raw_text: str | None = None,
        policy_engine: str | None = None,
        tool_events: list[dict[str, Any]] | None = None,
    ) -> Answer:
        return Answer(
            query=query,
            text=self.refusal_text,
            model_tag=self.model_tag,
            sources=list(hits),
            refused=True,
            refusal_reason=reason,
            raw_text=raw_text,
            policy_engine=policy_engine,
            tool_events=tool_events or [],
        )
