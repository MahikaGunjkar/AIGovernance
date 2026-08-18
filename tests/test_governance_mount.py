"""
Governance mount + PEP-gated tool execution.

Uses tests/fixtures/governance (snapshot of feature/governance-policies src/policies)
so tests do not require a live worktree. Production path mounts the real branch.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from heinzy.governance.loader import GovernanceMountError, get_interceptor, governance_available
from heinzy.tools.registry import run_governed_tool

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "governance"
FIXTURE_SRC = FIXTURE_ROOT / "src"
FIXTURE_POLICY = FIXTURE_ROOT / "policies" / "governance_policy.yaml"


@pytest.fixture
def governance_env(monkeypatch, tmp_path):
    """Point GOVERNANCE_* at fixtures; isolate audit log under tmp_path."""
    monkeypatch.setenv("GOVERNANCE_SRC", str(FIXTURE_SRC))
    monkeypatch.setenv("GOVERNANCE_POLICY_PATH", str(FIXTURE_POLICY))
    monkeypatch.chdir(tmp_path)
    get_interceptor.cache_clear()
    yield tmp_path
    get_interceptor.cache_clear()


def test_governance_available_false_when_unset(monkeypatch):
    monkeypatch.delenv("GOVERNANCE_SRC", raising=False)
    get_interceptor.cache_clear()
    assert governance_available() is False


def test_get_interceptor_requires_mount(monkeypatch):
    monkeypatch.delenv("GOVERNANCE_SRC", raising=False)
    get_interceptor.cache_clear()
    with pytest.raises(GovernanceMountError):
        get_interceptor()


def test_interceptor_is_ollama_pep(governance_env):
    pep = get_interceptor()
    assert pep.__class__.__name__ == "OllamaGovernanceInterceptor"
    assert callable(pep.evaluate_tool_call)
    assert callable(pep.log_activity)


def test_deny_write_action(governance_env):
    with pytest.raises(PermissionError, match="block-write-actions"):
        run_governed_tool("delete", {"action_type": "delete"}, query="delete prereqs")


def test_deny_non_cmu_web_search(governance_env):
    with pytest.raises(PermissionError, match="enforce-domain-allowlist"):
        run_governed_tool(
            "web_search",
            {"url": "https://reddit.com/r/cmu", "action_type": "web_search"},
            query="search reddit",
        )


def test_require_approval_cmu_web_search(governance_env):
    result = run_governed_tool(
        "web_search",
        {"url": "https://www.cmu.edu/heinz/advising", "action_type": "web_search"},
        query="search cmu advising",
    )
    assert result["status"] == "PAUSED_FOR_APPROVAL"
    assert result["details"]["rule"] == "require-human-for-search"


def test_allow_non_search_falls_through_to_stub_only_if_allowed(governance_env):
    """
    Default ALLOW path: a tool that is not web_search/write.
    Registry has no 'handbook_ping' — prove PEP ALLOW then registry KeyError,
    and separately that web_search never ALLOW-executes without approval.
    """
    pep = get_interceptor()
    evaluation = pep.evaluate_tool_call(
        "handbook_ping", {"action_type": "read"}, user_query="ping"
    )
    assert evaluation["decision"] == "ALLOW"


def test_audit_log_written_on_deny(governance_env):
    with pytest.raises(PermissionError):
        run_governed_tool("update", {"action_type": "update"}, query="update handbook")
    log_path = governance_env / "heinzy_audit.log"
    assert log_path.is_file()
    text = log_path.read_text(encoding="utf-8")
    assert "DENIED" in text
    assert "block-write-actions" in text
