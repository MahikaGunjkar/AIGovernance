"""
Layer 1 policy and Layer 2 separation tests (#11).

The two layers are tested apart because they fail apart. Layer 1 is a
structural decision an engine can replay and audit. Layer 2 is a model
judgement that has to be measured per model. The AGT tests skip cleanly when
the governance extra is not installed, so a plain checkout still runs green.
"""
from __future__ import annotations

import pytest

from heinzy.common.config import load_config
from heinzy.generation.abstain import detects_refusal, sentinel_key
from heinzy.generation.policy import (
    ENGINE_AGT,
    ENGINE_BUILTIN,
    REASON_ENGINE_UNAVAILABLE,
    REASON_NO_CONTEXT,
    BuiltinContextPolicy,
    DeniedPolicy,
    get_context_policy,
)

has_agt = True
try:  # pragma: no cover - depends on whether the extra is installed
    import agent_control_plane  # noqa: F401
except ImportError:  # pragma: no cover
    has_agt = False

needs_agt = pytest.mark.skipif(not has_agt, reason="governance extra not installed")


@pytest.fixture
def cfg():
    return load_config("config.yaml")


# --- Layer 1, builtin ------------------------------------------------------

def test_builtin_denies_with_no_context():
    d = BuiltinContextPolicy(min_hits=1).decide(0)
    assert not d.allowed
    assert d.reason == REASON_NO_CONTEXT
    assert d.engine == ENGINE_BUILTIN


def test_builtin_allows_once_min_hits_met():
    assert BuiltinContextPolicy(min_hits=2).decide(2).allowed
    assert not BuiltinContextPolicy(min_hits=2).decide(1).allowed


def test_engine_comes_from_config(cfg):
    cfg.generation.policy.engine = ENGINE_BUILTIN
    assert get_context_policy(cfg).engine == ENGINE_BUILTIN


def test_unknown_engine_is_an_error_not_a_default(cfg):
    cfg.generation.policy.engine = "something-else"
    with pytest.raises(NotImplementedError):
        get_context_policy(cfg)


# --- Layer 1, fail closed --------------------------------------------------

def test_missing_engine_denies_rather_than_falling_back():
    """The whole point of the fail-closed rule.

    A gate that quietly degrades to the builtin check when its dependency is
    missing still reports as governed while enforcing nothing anyone asked for.
    """
    denied = DeniedPolicy(REASON_ENGINE_UNAVAILABLE)
    d = denied.decide(99)  # plenty of context, still denied
    assert not d.allowed
    assert d.reason == REASON_ENGINE_UNAVAILABLE


def test_requesting_agt_without_the_extra_denies(cfg, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_agt(name, *args, **kwargs):
        if name.startswith("agent_control_plane"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_agt)
    cfg.generation.policy.engine = ENGINE_AGT
    policy = get_context_policy(cfg)

    assert isinstance(policy, DeniedPolicy)
    assert not policy.decide(50).allowed


# --- Layer 1, AGT ----------------------------------------------------------

@needs_agt
def test_agt_denies_without_context(cfg):
    cfg.generation.policy.engine = ENGINE_AGT
    d = get_context_policy(cfg).decide(0)
    assert not d.allowed
    assert d.engine == ENGINE_AGT


@needs_agt
def test_agt_allows_with_context(cfg):
    cfg.generation.policy.engine = ENGINE_AGT
    assert get_context_policy(cfg).decide(5).allowed


@needs_agt
def test_agt_denial_is_written_to_the_audit_log(cfg):
    """The ticket's QA. A denial has to be visible in the engine's own log."""
    cfg.generation.policy.engine = ENGINE_AGT
    d = get_context_policy(cfg).decide(0)

    denials = [e for e in d.audit if e.get("event_type") == "request_denied"]
    assert denials, f"no denial recorded, log was {d.audit}"
    assert denials[-1]["details"]["reason"] == "policy_violation"


@needs_agt
def test_agt_registers_the_rule_in_the_audit_log(cfg):
    cfg.generation.policy.engine = ENGINE_AGT
    d = get_context_policy(cfg).decide(0)
    added = [e for e in d.audit if e.get("event_type") == "policy_added"]
    assert any(
        e["details"]["rule_id"] == "heinzy-require-retrieved-context" for e in added
    )


# --- Layer 2, separated ----------------------------------------------------

def test_layer_two_lives_apart_from_layer_one():
    """Importable and usable with no config, no engine and no model."""
    assert detects_refusal("INSUFFICIENT_CONTEXT", "INSUFFICIENT_CONTEXT")
    assert not detects_refusal("Students take 54 units.", "INSUFFICIENT_CONTEXT")


def test_sentinel_key_strips_formatting():
    assert sentinel_key("**INSUFFICIENT CONTEXT**") == "INSUFFICIENTCONTEXT"


def test_empty_sentinel_never_matches():
    """A blank sentinel must not turn every answer into a refusal."""
    assert not detects_refusal("any answer at all", "")
