"""
Layer 1 of abstention as a policy decision. See issue #11.

The question Layer 1 asks is structural. Did retrieval return enough context to
ground an answer. That is exactly the shape a policy engine evaluates, allow or
deny on a field before the next step runs, so it belongs in one rather than in a
bare `if` buried in the generator.

Two engines implement the same decision.

`builtin` is the deterministic count check. No dependencies, always available,
and it is what runs on a plain `pip install -e .`.

`agt` routes the same condition through the Microsoft Agent Governance Toolkit,
registering it as a real PolicyRule whose validator receives the request and
returns False to deny. The decision is then visible in the kernel's audit log
alongside every other governed action, which is the point of using an engine at
all. Install with `pip install -e ".[governance]"`.

pre:  cfg carries generation.abstain.min_hits and optionally generation.policy
post: decide() returns a PolicyDecision saying whether generation may proceed,
      which engine decided, and an audit trail when the engine keeps one
invariant: FAIL CLOSED. If the configured engine cannot be loaded or raises,
           the decision is deny, never allow. A safety gate that silently
           degrades to "yes" when its dependency is missing is worse than the
           `if` it replaced, because it still looks governed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ENGINE_BUILTIN = "builtin"
ENGINE_AGT = "agt"

REASON_ALLOWED = "sufficient_context"
REASON_NO_CONTEXT = "no_retrieved_context"
REASON_ENGINE_UNAVAILABLE = "policy_engine_unavailable"
REASON_ENGINE_ERROR = "policy_engine_error"


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    engine: str
    audit: list[dict[str, Any]] = field(default_factory=list)


class BuiltinContextPolicy:
    """Deterministic count check. The always-available path."""

    engine = ENGINE_BUILTIN

    def __init__(self, min_hits: int) -> None:
        self.min_hits = min_hits

    def decide(self, hit_count: int) -> PolicyDecision:
        if hit_count < self.min_hits:
            return PolicyDecision(False, REASON_NO_CONTEXT, self.engine)
        return PolicyDecision(True, REASON_ALLOWED, self.engine)


class AgtContextPolicy:
    """The same condition, evaluated by the toolkit and written to its audit log.

    Generation is modelled as an API_CALL, which is what it is, since answering
    means calling the model host over HTTP. The agent is granted exactly the
    permission that action needs and nothing more, so a denial here is always a
    policy denial rather than a permissions accident.
    """

    engine = ENGINE_AGT
    rule_id = "heinzy-require-retrieved-context"

    def __init__(self, min_hits: int, agent_id: str = "heinzy-advisor") -> None:
        from datetime import datetime

        from agent_control_plane.agent_kernel import (
            ActionType,
            AgentContext,
            AgentKernel,
            PermissionLevel,
            PolicyRule,
        )

        self.min_hits = min_hits
        self._ActionType = ActionType
        self._kernel = AgentKernel()
        self._context = AgentContext(
            agent_id=agent_id,
            session_id=f"{agent_id}-session",
            created_at=datetime(2026, 1, 1),
            permissions={ActionType.API_CALL: PermissionLevel.READ_WRITE},
        )
        self._kernel.add_policy_rule(
            PolicyRule(
                rule_id=self.rule_id,
                name="Require retrieved context before answering",
                description=(
                    "Deny answer generation unless retrieval returned at least "
                    "min_hits chunks that cleared retrieval.score_floor. Without "
                    "context there is nothing to ground an answer in, so calling "
                    "the model can only invent one."
                ),
                action_types=[ActionType.API_CALL],
                # The toolkit reads True as allow and False as deny.
                validator=lambda req: int(req.parameters.get("hit_count", 0)) >= self.min_hits,
                priority=100,
            )
        )

    def decide(self, hit_count: int) -> PolicyDecision:
        request = self._kernel.submit_request(
            agent_context=self._context,
            action_type=self._ActionType.API_CALL,
            parameters={"hit_count": hit_count, "min_hits": self.min_hits},
        )
        allowed = request.status.value == "approved"
        return PolicyDecision(
            allowed=allowed,
            reason=REASON_ALLOWED if allowed else REASON_NO_CONTEXT,
            engine=self.engine,
            audit=self._kernel.get_audit_log(),
        )


class DeniedPolicy:
    """Stand-in used when the configured engine could not be loaded.

    Every decision is deny. This is the fail-closed path, and it reports why so
    the refusal is not mistaken for a genuine lack of context.
    """

    engine = ENGINE_AGT

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def decide(self, hit_count: int) -> PolicyDecision:
        return PolicyDecision(False, self.reason, self.engine)


def get_context_policy(cfg):
    """Build the Layer 1 policy named by generation.policy.engine.

    Defaults to builtin so a plain install keeps working. Requesting agt without
    the extra installed yields a policy that denies rather than one that quietly
    falls back to builtin, because a silent downgrade would mean the governance
    the config asked for was not running and nobody was told.
    """
    generation = getattr(cfg, "generation", None)
    abstain = getattr(generation, "abstain", None)
    min_hits = int(getattr(abstain, "min_hits", 1) or 1)

    policy = getattr(generation, "policy", None)
    engine = (getattr(policy, "engine", None) or ENGINE_BUILTIN).strip().lower()
    agent_id = getattr(policy, "agent_id", None) or "heinzy-advisor"

    if engine == ENGINE_BUILTIN:
        return BuiltinContextPolicy(min_hits)
    if engine == ENGINE_AGT:
        try:
            return AgtContextPolicy(min_hits, agent_id=agent_id)
        except ImportError:
            return DeniedPolicy(REASON_ENGINE_UNAVAILABLE)
        except Exception:
            return DeniedPolicy(REASON_ENGINE_ERROR)
    raise NotImplementedError(
        f"generation.policy.engine={engine!r} has no implementation. "
        f"Use {ENGINE_BUILTIN!r} or {ENGINE_AGT!r}."
    )
