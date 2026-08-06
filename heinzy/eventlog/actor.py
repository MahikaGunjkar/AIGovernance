"""
Actor identity for audit records (task A5).

Who initiated the retrieval (or, later, generation / governance action).
Call sites pass an Actor into Retriever.retrieve(); the event log stamps it
onto every envelope. No secrets — use opaque staff IDs, not passwords/tokens.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Actor:
    actor_id: str
    role: str = "advisor"  # advisor | student | system | … (free string for now)

    def __post_init__(self) -> None:
        if not self.actor_id or not str(self.actor_id).strip():
            raise ValueError("actor_id must be a non-empty string")
        if not self.role or not str(self.role).strip():
            raise ValueError("role must be a non-empty string")
        object.__setattr__(self, "actor_id", str(self.actor_id).strip())
        object.__setattr__(self, "role", str(self.role).strip())

    def to_dict(self) -> dict[str, str]:
        return {"actor_id": self.actor_id, "role": self.role}
