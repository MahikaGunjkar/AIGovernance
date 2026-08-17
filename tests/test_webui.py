"""Web UI routing + decline/source rendering (ticket #17). No live infra needed
— uses a fake engine mirroring the real Retriever/Generator return shapes."""
from dataclasses import dataclass

import pytest

from heinzy.webui.app import _Engine, _looks_declined, create_app


@dataclass
class _FakeChunk:
    section_path: str
    source_pages: list
    score: float


@dataclass
class _FakeResult:
    query: str
    hits: list


@dataclass
class _FakeAnswer:
    query: str
    text: str
    model_tag: str
    sources: list


class _FakeRetriever:
    def retrieve(self, q, k=None):
        return _FakeResult(q, [_FakeChunk("MISM > Electives", [21], 0.83)])


class _FakeGenerator:
    def generate(self, q, hits):
        if "weather" in q.lower():
            return _FakeAnswer(q, "I cannot answer; not covered by the handbook.", "gemma2:9b", hits)
        return _FakeAnswer(q, "MISM students may take up to 4 electives.", "gemma2:9b", hits)


@pytest.fixture
def client():
    eng = _Engine(_FakeRetriever(), _FakeGenerator(), "gemma2:9b", 42)
    return create_app(eng).test_client()


def test_decline_detector():
    assert _looks_declined("I cannot answer that.")
    assert not _looks_declined("MISM needs 144 units.")


def test_health(client):
    h = client.get("/api/health").get_json()
    assert h["ready"] and h["chunks_indexed"] == 42


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Heinzy" in r.data


def test_answered_turn_has_sources(client):
    a = client.post("/api/ask", json={"question": "How many electives?"}).get_json()
    assert a["declined"] is False
    assert a["sources"][0]["section_path"] == "MISM > Electives"
    assert a["sources"][0]["source_pages"] == [21]


def test_declined_turn_flagged(client):
    d = client.post("/api/ask", json={"question": "What's the weather?"}).get_json()
    assert d["declined"] is True


def test_empty_question_rejected(client):
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
