"""
Event log tests (A5): append-only JSONL retrieval audit records + actor.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heinzy.common.config import load_config
from heinzy.eventlog import Actor
from heinzy.eventlog.writer import JsonlEventLog, get_event_log, iter_records
from heinzy.retrieval.embedder import Embedder
from heinzy.retrieval.retrieve import Retriever
from heinzy.retrieval.store import InMemoryStore, StoredChunk

CHUNKS = [
    ("c1", "graduation requires 144 units", "H > Grad", [1]),
    ("c2", "core includes statistics", "H > Core", [2]),
]

ADVISOR = Actor(actor_id="advisor-jsmith", role="advisor")


@pytest.fixture
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def seeded_store(cfg):
    emb = Embedder(cfg.embed.model_tag, cfg.embed.dimension)
    store = InMemoryStore()
    store.add([
        StoredChunk(cid, emb.embed(text), text, "doc-x", sec, pages)
        for cid, text, sec, pages in CHUNKS
    ])
    return store


def test_append_retrieval_writes_one_json_line(cfg, seeded_store, tmp_path: Path):
    log_path = tmp_path / "retrieval.jsonl"
    event_log = JsonlEventLog(log_path)
    r = Retriever(cfg, store=seeded_store, event_log=event_log)

    result = r.retrieve("units", k=1, actor=ADVISOR)
    assert result.audit_record is not None
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event_type"] == "retrieval"
    assert rec["event_id"]
    assert rec["ts"]
    assert rec["actor"] == {"actor_id": "advisor-jsmith", "role": "advisor"}
    assert rec["query"] == "units"
    assert rec["config_hash"] == cfg.config_hash
    assert isinstance(rec["hits"], list)
    assert len(rec["hits"]) <= 1


def test_append_is_append_only(cfg, seeded_store, tmp_path: Path):
    log_path = tmp_path / "retrieval.jsonl"
    event_log = JsonlEventLog(log_path)
    r = Retriever(cfg, store=seeded_store, event_log=event_log)

    r.retrieve("units", k=1, actor=ADVISOR)
    r.retrieve("statistics", k=1, actor=Actor("advisor-lee", role="advisor"))

    records = list(iter_records(log_path))
    assert len(records) == 2
    assert records[0]["event_id"] != records[1]["event_id"]
    assert records[0]["actor"]["actor_id"] == "advisor-jsmith"
    assert records[1]["actor"]["actor_id"] == "advisor-lee"


def test_default_actor_used_when_per_call_omitted(cfg, seeded_store, tmp_path: Path):
    log_path = tmp_path / "retrieval.jsonl"
    event_log = JsonlEventLog(log_path)
    r = Retriever(
        cfg, store=seeded_store, event_log=event_log, default_actor=ADVISOR
    )
    result = r.retrieve("units", k=1)
    assert result.audit_record["actor"]["actor_id"] == "advisor-jsmith"


def test_logging_without_actor_raises(cfg, seeded_store, tmp_path: Path):
    event_log = JsonlEventLog(tmp_path / "retrieval.jsonl")
    r = Retriever(cfg, store=seeded_store, event_log=event_log)
    with pytest.raises(ValueError, match="actor is required"):
        r.retrieve("units", k=1)


def test_empty_actor_id_rejected():
    with pytest.raises(ValueError):
        Actor(actor_id="   ", role="advisor")


def test_disabled_log_does_not_write(cfg, seeded_store, tmp_path: Path):
    log_path = tmp_path / "retrieval.jsonl"
    event_log = JsonlEventLog(log_path, enabled=False)
    r = Retriever(cfg, store=seeded_store, event_log=event_log)

    result = r.retrieve("units", k=1, actor=ADVISOR)
    assert result.audit_record is not None  # record still built
    assert result.audit_record["actor"]["role"] == "advisor"
    assert not log_path.exists()


def test_no_event_log_leaves_audit_record_none(cfg, seeded_store):
    r = Retriever(cfg, store=seeded_store)
    result = r.retrieve("units", k=1)
    assert result.audit_record is None


def test_get_event_log_from_config(cfg):
    log = get_event_log(cfg)
    assert log is not None
    assert log.enabled is True
    assert str(log.path).replace("\\", "/").endswith("data/logs/retrieval.jsonl")
