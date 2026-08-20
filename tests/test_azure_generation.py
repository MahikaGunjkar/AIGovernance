"""
Azure OpenAI / AI Foundry generation path (chat + governed tools).

Offline: stubs requests.post with Azure Chat Completions shapes. Overrides the
autouse Ollama provider fixture so these tests actually hit _chat_azure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heinzy.common.config import load_config
from heinzy.generation import generator as generator_module
from heinzy.generation.generator import Generator
from heinzy.governance.loader import get_interceptor
from heinzy.retrieval.store import ScoredChunk

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "governance"
FIXTURE_SRC = FIXTURE_ROOT / "src"
FIXTURE_POLICY = FIXTURE_ROOT / "policies" / "governance_policy.yaml"

HITS = [
    ScoredChunk(
        chunk_id="c1",
        text="Students must complete seven core courses totalling 144 units.",
        score=0.81,
        doc_id="doc-x",
        section_path="Handbook > 4. Curriculum > 4.1. Core Courses",
        source_pages=[11],
    ),
]


class _AzureResponse:
    def __init__(self, body: dict, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise generator_module.requests.HTTPError(f"{self.status_code}")

    def json(self) -> dict:
        return self._body


def _azure_message(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "azure_openai")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT", "https://test.services.ai.azure.com"
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deploy")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    monkeypatch.delenv("GOVERNANCE_SRC", raising=False)
    monkeypatch.delenv("GOVERNANCE_POLICY_PATH", raising=False)
    get_interceptor.cache_clear()
    yield
    get_interceptor.cache_clear()


@pytest.fixture
def azure_governance_env(azure_env, monkeypatch, tmp_path):
    monkeypatch.setenv("GOVERNANCE_SRC", str(FIXTURE_SRC))
    monkeypatch.setenv("GOVERNANCE_POLICY_PATH", str(FIXTURE_POLICY))
    monkeypatch.chdir(tmp_path)
    get_interceptor.cache_clear()
    yield tmp_path
    get_interceptor.cache_clear()


def test_azure_plain_chat_sends_foundry_url_and_api_key(cfg, azure_env, monkeypatch):
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None, headers=None, **kwargs):
        calls.append({"url": url, "payload": json, "headers": headers})
        return _AzureResponse(
            _azure_message('Seven cores (see "4.1. Core Courses").')
        )

    monkeypatch.setattr(generator_module.requests, "post", fake_post)
    gen = Generator(cfg)
    assert gen.provider == "azure_openai"
    assert not gen.use_tools
    answer = gen.generate("core requirements?", HITS)

    assert not answer.refused
    assert answer.model_tag == "test-deploy"
    assert calls[0]["url"].endswith("/openai/v1/chat/completions")
    assert calls[0]["headers"]["api-key"] == "test-key"
    assert calls[0]["payload"]["model"] == "test-deploy"
    assert "tools" not in calls[0]["payload"]


def test_azure_use_tools_when_governance_mounted(cfg, azure_governance_env, monkeypatch):
    deny_args = json.dumps({"action_type": "delete"})
    responses = [
        _AzureResponse(
            _azure_message(
                content=None,
                tool_calls=[
                    {
                        "id": "call_deny_1",
                        "type": "function",
                        "function": {
                            "name": "delete",
                            "arguments": deny_args,
                        },
                    }
                ],
            )
        ),
        _AzureResponse(
            _azure_message('Seven cores (see "4.1. Core Courses").')
        ),
    ]
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None, headers=None, **kwargs):
        calls.append({"url": url, "payload": json, "headers": headers})
        return responses.pop(0)

    monkeypatch.setattr(generator_module.requests, "post", fake_post)
    gen = Generator(cfg)
    assert gen.use_tools is True
    answer = gen.generate("core requirements?", HITS)

    assert not answer.refused
    assert answer.tool_events and answer.tool_events[0]["status"] == "DENIED"
    # First turn advertises tools; second is the follow-up after the deny.
    assert "tools" in calls[0]["payload"]
    assert any(t["function"]["name"] == "web_search" for t in calls[0]["payload"]["tools"])
    tool_msgs = [m for m in calls[1]["payload"]["messages"] if m.get("role") == "tool"]
    assert tool_msgs
    assert tool_msgs[0]["tool_call_id"] == "call_deny_1"
    assert "DENIED" in tool_msgs[0]["content"]


def test_azure_tool_call_pauses_for_approval(cfg, azure_governance_env, monkeypatch):
    search_args = json.dumps({"url": "https://www.cmu.edu/heinz/advising"})

    def fake_post(url, json=None, timeout=None, headers=None, **kwargs):
        return _AzureResponse(
            _azure_message(
                content=None,
                tool_calls=[
                    {
                        "id": "call_search_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": search_args,
                        },
                    }
                ],
            )
        )

    monkeypatch.setattr(generator_module.requests, "post", fake_post)
    answer = Generator(cfg).generate("find advising page", HITS)

    assert answer.paused_for_approval
    assert answer.tool_events[0]["status"] == "PAUSED_FOR_APPROVAL"


def test_messages_for_azure_round_trips_tool_calls():
    shared = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"url": "https://www.cmu.edu"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"status": "DENIED"}',
        },
    ]
    azure = Generator._messages_for_azure(shared)
    assert azure[2]["tool_calls"][0]["id"] == "call_1"
    assert azure[2]["content"] is None
    assert azure[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"status": "DENIED"}',
    }
