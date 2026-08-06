"""
Append-only JSONL event log (Prototype task A5).

pre:  path's parent is writable (created on first append if needed);
      append_retrieval() receives a non-empty Actor
post: each append_retrieval() writes one JSON line and returns the full record
invariant: records are append-only; path and enabled come from config, not
           hardcoded callers (except tests injecting a temp path).

Envelope fields (event_id, event_type, ts, actor) wrap the retrieval payload
from RetrievalResult.to_log_record() so generation/governance events can share
the same log later with a different event_type.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from heinzy.eventlog.actor import Actor

if TYPE_CHECKING:
    from heinzy.common.config import Config
    from heinzy.retrieval.retrieve import RetrievalResult

EVENT_TYPE_RETRIEVAL = "retrieval"


class JsonlEventLog:
    """Append-only JSONL writer for audit records."""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    def append_retrieval(
        self,
        result: RetrievalResult,
        actor: Actor,
    ) -> dict[str, Any]:
        """Build a retrieval audit record and append it when enabled."""
        record = self.build_retrieval_record(result, actor)
        if self.enabled:
            self._append_line(record)
        return record

    def build_retrieval_record(
        self,
        result: RetrievalResult,
        actor: Actor,
    ) -> dict[str, Any]:
        """Envelope + actor + A2 payload. Does not touch disk."""
        if not isinstance(actor, Actor):
            raise TypeError("actor must be an Actor instance")
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": EVENT_TYPE_RETRIEVAL,
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor.to_dict(),
            **result.to_log_record(),
        }

    def _append_line(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Read every JSON object from the log (empty list if missing)."""
        return list(iter_records(self.path))


def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_event_log(cfg: Config) -> JsonlEventLog | None:
    """Factory from config.event_log. Returns None when section missing/disabled."""
    section = getattr(cfg, "event_log", None)
    if section is None:
        return None
    if not getattr(section, "enabled", True):
        return None
    path = getattr(section, "path", None)
    if not path:
        raise ValueError("event_log.path must be set in config.yaml when enabled")
    return JsonlEventLog(path=path, enabled=True)
