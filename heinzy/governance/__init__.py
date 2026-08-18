"""Governance mount: load OllamaGovernanceInterceptor from GOVERNANCE_SRC."""

from heinzy.governance.loader import get_interceptor, governance_available

__all__ = ["get_interceptor", "governance_available"]
