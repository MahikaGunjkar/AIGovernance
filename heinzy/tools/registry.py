"""
Tool runners that sit *behind* OllamaGovernanceInterceptor.

PEP first (evaluate_tool_call / log_activity), then dispatch. Write tools exist
only so policy can DENY them; web_search is a Phase-1 stub (no live HTTP).
"""
from __future__ import annotations

from typing import Any, Callable

from heinzy.governance.loader import get_interceptor

# Ollama / OpenAI-style tool schemas advertised to the model when governance is on.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search or fetch an official university web page. "
                "Only CMU domains are in policy scope; others are denied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to search or fetch (e.g. https://www.cmu.edu/...).",
                    },
                    "action_type": {
                        "type": "string",
                        "description": "Optional; defaults to web_search.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create",
            "description": "Create or write a record. Phase 1: always denied by governance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update",
            "description": "Update a record. Phase 1: always denied by governance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": "Delete a record. Phase 1: always denied by governance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert",
            "description": "Insert a record. Phase 1: always denied by governance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "data": {"type": "object"},
                },
            },
        },
    },
]


def _stub_web_search(tool_args: dict[str, Any]) -> dict[str, Any]:
    url = tool_args.get("url", "")
    return {
        "status": "SUCCESS",
        "tool": "web_search",
        "url": url,
        "snippet": (
            f"[stub] No live HTTP in Phase 1. Would fetch allowlisted URL: {url}"
        ),
    }


def _deny_only_write(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError(
        f"Write tool '{tool_name}' must never execute; governance should have DENY'd it. "
        f"args={tool_args!r}"
    )


_RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "web_search": _stub_web_search,
    "create": lambda args: _deny_only_write("create", args),
    "update": lambda args: _deny_only_write("update", args),
    "delete": lambda args: _deny_only_write("delete", args),
    "insert": lambda args: _deny_only_write("insert", args),
}


def execute(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch after ALLOW. Unknown tools raise KeyError."""
    runner = _RUNNERS.get(tool_name)
    if runner is None:
        raise KeyError(f"No tool runner registered for {tool_name!r}")
    return runner(tool_args)


def run_governed_tool(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    *,
    query: str = "",
    agent_id: str = "heinzy-advisor",
) -> dict[str, Any]:
    """
    Mount point in front of tool execution: PEP then registry.

    Uses OllamaGovernanceInterceptor.evaluate_tool_call / log_activity from the
    GOVERNANCE_SRC mount. Does not reimplement policy decisions.
    """
    tool_args = dict(tool_args or {})
    # Ensure action_type is visible to the PEP for write tools named create/etc.
    if "action_type" not in tool_args:
        tool_args["action_type"] = tool_name

    pep = get_interceptor()
    evaluation = pep.evaluate_tool_call(tool_name, tool_args, user_query=query)
    decision = evaluation.get("decision", "DENY")

    if decision == "DENY":
        pep.log_activity(agent_id, query, tool_name, tool_args, evaluation, "DENIED")
        raise PermissionError(
            f"[AGT DENIED] Rule: {evaluation.get('rule')} - {evaluation.get('reason')}"
        )

    if decision == "REQUIRE_APPROVAL":
        pep.log_activity(
            agent_id, query, tool_name, tool_args, evaluation, "PAUSED_FOR_APPROVAL"
        )
        return {"status": "PAUSED_FOR_APPROVAL", "details": evaluation}

    pep.log_activity(agent_id, query, tool_name, tool_args, evaluation, "SUCCESS")
    result = execute(tool_name, tool_args)
    return result
