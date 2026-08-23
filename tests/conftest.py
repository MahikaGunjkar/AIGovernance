"""Test defaults: keep unit tests on the Ollama path.

A developer .env with MODEL_PROVIDER=azure_openai must not change contract
tests that stub requests.post as an Ollama /api/chat call.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_ollama_provider_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    for key in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(key, raising=False)
