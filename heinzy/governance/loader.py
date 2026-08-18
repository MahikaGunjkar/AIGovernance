"""
Load the PEP (OllamaGovernanceInterceptor) from a Docker/worktree mount.

The class lives on feature/governance-policies (src/ollama_governance_interceptor.py).
This module only puts GOVERNANCE_SRC on sys.path and constructs the interceptor
with GOVERNANCE_POLICY_PATH. It does not reimplement policy logic.
"""
from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


class GovernanceMountError(RuntimeError):
    """Raised when the governance mount is missing or unloadable."""


def _governance_src() -> Path:
    raw = os.environ.get("GOVERNANCE_SRC", "").strip()
    if not raw:
        raise GovernanceMountError(
            "GOVERNANCE_SRC is unset. Mount feature/governance-policies src "
            "(e.g. GOVERNANCE_SRC=/governance/src) or leave governance disabled."
        )
    path = Path(raw)
    if not path.is_dir():
        raise GovernanceMountError(
            f"GOVERNANCE_SRC does not exist or is not a directory: {path}"
        )
    return path.resolve()


def _policy_path() -> str:
    raw = os.environ.get("GOVERNANCE_POLICY_PATH", "").strip()
    if raw:
        return raw
    # Sensible default when only GOVERNANCE_SRC is set (sibling policies/).
    sibling = _governance_src().parent / "policies" / "governance_policy.yaml"
    if sibling.is_file():
        return str(sibling)
    raise GovernanceMountError(
        "GOVERNANCE_POLICY_PATH is unset and no sibling policies/governance_policy.yaml found."
    )


def governance_available() -> bool:
    """True when GOVERNANCE_SRC points at a directory containing the interceptor module."""
    raw = os.environ.get("GOVERNANCE_SRC", "").strip()
    if not raw:
        return False
    src = Path(raw)
    return src.is_dir() and (src / "ollama_governance_interceptor.py").is_file()


@lru_cache(maxsize=1)
def get_interceptor() -> Any:
    """
    Import and construct OllamaGovernanceInterceptor from the mounted tree.

    Cached for process lifetime; clear with get_interceptor.cache_clear() in tests.
    """
    src = _governance_src()
    module_file = src / "ollama_governance_interceptor.py"
    if not module_file.is_file():
        raise GovernanceMountError(
            f"Mounted governance src missing ollama_governance_interceptor.py: {src}"
        )

    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    # Fresh import if a previous test left a stale module.
    if "ollama_governance_interceptor" in sys.modules:
        mod = importlib.reload(sys.modules["ollama_governance_interceptor"])
    else:
        mod = importlib.import_module("ollama_governance_interceptor")

    cls = getattr(mod, "OllamaGovernanceInterceptor", None)
    if cls is None:
        raise GovernanceMountError(
            "Mounted ollama_governance_interceptor.py has no OllamaGovernanceInterceptor"
        )
    return cls(policy_path=_policy_path())
