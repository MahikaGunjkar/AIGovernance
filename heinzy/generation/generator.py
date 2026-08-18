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

When GOVERNANCE_SRC is set (Docker/worktree mount), the generate() path may
enter an Ollama tool-calling loop. Every tool_call is gated by the mounted
OllamaGovernanceInterceptor before heinzy.tools runners execute.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from heinzy.governance.loader import governance_available
from heinzy.retrieval.store import ScoredChunk
from heinzy.tools.registry import TOOL_DEFINITIONS, run_governed_tool

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

SYSTEM_PROMPT_WITH_TOOLS = (
    SYSTEM_PROMPT
    + " You may call tools when needed. Write/create/update/delete/insert actions "
    "are forbidden. Web search is only for official CMU domains and may require "
    "human approval. Prefer handbook excerpts over tools when they suffice."
)


@dataclass
class Answer:
    query: str
    text: str
    model_tag: str
    sources: list[ScoredChunk]
    paused_for_approval: bool = False
    tool_events: list[dict[str, Any]] = field(default_factory=list)


class Generator:
    def __init__(self, cfg) -> None:
        self.model_tag = os.environ.get("MODEL_TAG") or cfg.model.tag
        self.endpoint = (getattr(cfg.model, "endpoint", "") or _DEFAULT_ENDPOINT).rstrip("/")
        gov = getattr(cfg, "governance", None)
        self._governance_cfg = gov
        # Enable tool loop when mount is present and config does not disable it.
        enabled = True if gov is None else bool(getattr(gov, "enabled", True))
        self.use_tools = bool(enabled and governance_available())
        self.max_tool_rounds = int(getattr(gov, "max_tool_rounds", 3) if gov else 3)
        self.agent_id = str(getattr(gov, "agent_id", "heinzy-advisor") if gov else "heinzy-advisor")

    def _chat(self, messages: list[dict[str, Any]], *, tools: list[dict] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_tag,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if tools:
            payload["tools"] = tools
        resp = requests.post(
            f"{self.endpoint}/api/chat",
            json=payload,
            timeout=180,
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

    def generate(self, query: str, hits: list[ScoredChunk]) -> Answer:
        context = "\n\n".join(
            f"[{h.section_path}] (p{h.source_pages}): {h.text}" for h in hits
        )
        user_prompt = f"Handbook excerpts:\n\n{context}\n\nQuestion: {query}"

        if not self.use_tools:
            data = self._chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
            )
            return Answer(
                query=query,
                text=data["message"]["content"],
                model_tag=self.model_tag,
                sources=hits,
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOLS},
            {"role": "user", "content": user_prompt},
        ]
        tool_events: list[dict[str, Any]] = []

        for _ in range(self.max_tool_rounds):
            data = self._chat(messages, tools=TOOL_DEFINITIONS)
            message = data.get("message") or {}
            messages.append(message)

            tool_calls = self._tool_calls_from_message(message)
            if not tool_calls:
                return Answer(
                    query=query,
                    text=message.get("content") or "",
                    model_tag=self.model_tag,
                    sources=hits,
                    tool_events=tool_events,
                )

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
                    return Answer(
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

                tool_events.append(
                    {"tool": name, "args": args, "status": "SUCCESS", "result": result}
                )
                messages.append(
                    {"role": "tool", "content": json.dumps(result)},
                )

        # Exhausted rounds — ask once more without tools for a final answer.
        data = self._chat(messages, tools=None)
        message = data.get("message") or {}
        return Answer(
            query=query,
            text=message.get("content") or "",
            model_tag=self.model_tag,
            sources=hits,
            tool_events=tool_events,
        )
