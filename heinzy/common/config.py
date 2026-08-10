"""
Config loader (Infra task S5).

Loads config.yaml, expands ${ENV_VAR} references from the environment, and
computes a stable config_hash. The hash goes into every eval result and every
event-log record so a run can be reproduced exactly.

Usage:
    from heinzy.common.config import load_config
    cfg = load_config()          # reads ./config.yaml
    cfg.retrieval.k              # -> 5
    cfg.config_hash              # -> "a1b2c3..." (12 hex chars)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively replace ${VAR} with os.environ[VAR] (empty string if unset)."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class _Section:
    """Dot-access wrapper over a dict, so cfg.retrieval.k works."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, val in data.items():
            setattr(self, key, _Section(val) if isinstance(val, dict) else val)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"_Section({self.__dict__!r})"


@dataclass
class Config:
    raw: dict[str, Any]
    config_hash: str

    def __getattr__(self, name: str) -> Any:
        # Delegates cfg.retrieval / cfg.chunk / ... to the parsed sections.
        try:
            return self._sections[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc


def _compute_hash(raw: dict[str, Any]) -> str:
    """Stable hash of config BEFORE env expansion.

    Secrets/endpoints come from env and must NOT change the hash, so we hash
    the file's declared structure, not the resolved values.
    """
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load .env into os.environ if present. Does not override existing vars."""
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path.resolve()}. "
            "Run from the repo root, or pass an explicit path."
        )
    # S7: pick up MODEL_ENDPOINT / CHROMA_HOST from local .env when unset.
    _load_dotenv(path.parent / ".env" if path.parent != Path(".") else Path(".env"))
    # Also try cwd .env (common when config path is absolute or nested).
    _load_dotenv(Path(".env"))

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config_hash = _compute_hash(raw)
    resolved = _expand_env(raw)

    cfg = Config(raw=raw, config_hash=config_hash)
    cfg._sections = {  # type: ignore[attr-defined]
        k: (_Section(v) if isinstance(v, dict) else v) for k, v in resolved.items()
    }
    return cfg


if __name__ == "__main__":
    c = load_config()
    print(f"config version : {c.raw.get('version')}")
    print(f"config_hash    : {c.config_hash}")
    print(f"retrieval.k    : {c.retrieval.k}")
    print(f"embed.model_tag: {c.embed.model_tag}")
    print(f"store.backend  : {c.vector_store.backend}")
