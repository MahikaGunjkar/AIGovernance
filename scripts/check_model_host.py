"""
Check that the configured generation host is reachable and serving our model.

Supports:
  - MODEL_PROVIDER=ollama (default): GET {MODEL_ENDPOINT}/api/tags
  - MODEL_PROVIDER=azure_openai: GET {endpoint}/openai/models?api-version=...

pre:  config.yaml is readable; provider env vars are set
post: exits 0 when the host answers and looks correct, non-zero otherwise
invariant: reads endpoint/tag from config + env, never hardcoded

Run from repo root:
    python scripts/check_model_host.py
    MODEL_ENDPOINT=http://localhost:11434 python scripts/check_model_host.py
"""
from __future__ import annotations

import sys

import requests

from heinzy.common.config import load_config
from heinzy.generation.generator import Generator


def _check_ollama(gen: Generator) -> int:
    print(f"provider  ollama")
    print(f"endpoint  {gen.endpoint}")
    print(f"wanted    {gen.model_tag}")
    print(f"auth      {'basic auth from MODEL_BASIC_AUTH' if gen.auth else 'none'}")

    try:
        resp = requests.get(f"{gen.endpoint}/api/tags", timeout=10, auth=gen.auth)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"\nUNREACHABLE. {type(exc).__name__} talking to {gen.endpoint}")
        print("Check that Ollama is running and MODEL_ENDPOINT in .env matches "
              "(default http://localhost:11434). For LAN/Docker hosts, use that "
              "machine's URL and confirm the port is reachable.")
        return 2

    tags = [m.get("name", "") for m in resp.json().get("models", [])]
    print(f"serving   {', '.join(tags) if tags else '(nothing)'}")

    if gen.model_tag in tags:
        print(f"\nOK. {gen.model_tag} is available at {gen.endpoint}")
        return 0

    print(f"\nMISMATCH. The host is up but does not serve {gen.model_tag}.")
    print("Either pull that tag on the host, or align config.yaml model.tag "
          "with what the host actually serves. Results stamp the tag they were "
          "generated with, so a mismatch here makes every recorded number "
          "misleading.")
    return 3


def _check_azure(gen: Generator) -> int:
    print(f"provider  azure_openai")
    print(f"endpoint  {gen.azure_endpoint}")
    print(f"deployment {gen.azure_deployment}")
    if gen.azure_v1_chat_url:
        print(f"chat_url  {gen.azure_v1_chat_url}")
        print("style     foundry_v1 (model in body)")
        # Foundry often has no /openai/models list; ping chat instead.
        try:
            data = gen._chat_azure(
                [{"role": "user", "content": "Reply with OK"}],
                tools=None,
            )
            text = (data.get("message") or {}).get("content") or ""
            print(f"ping      {text[:80]!r}")
            print(f"\nOK. Foundry chat reachable; model={gen.azure_deployment}")
            return 0
        except Exception as exc:
            print(f"\nUNREACHABLE. {type(exc).__name__}: {exc}")
            print("Check AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY (for THIS resource),")
            print("and AZURE_OPENAI_DEPLOYMENT / model name.")
            return 2

    print(f"api_ver   {gen.azure_api_version}")
    url = (
        f"{gen.azure_endpoint}/openai/models"
        f"?api-version={gen.azure_api_version}"
    )
    try:
        resp = requests.get(
            url,
            headers={"api-key": gen.azure_api_key},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"\nUNREACHABLE. {type(exc).__name__} talking to Azure OpenAI.")
        print("Check AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and network access.")
        return 2

    models = resp.json().get("data") or []
    ids = [m.get("id", "") for m in models if isinstance(m, dict)]
    print(f"models    {', '.join(ids[:12]) if ids else '(none listed)'}"
          + (" ..." if len(ids) > 12 else ""))

    if ids and gen.azure_deployment not in ids and not any(
        gen.azure_deployment in mid for mid in ids
    ):
        print(
            f"\nNOTE. Deployment {gen.azure_deployment!r} was not in /models "
            "(common — deployment name can differ from model id). "
            "Credentials look OK; try a real ask next."
        )
        return 0

    print(f"\nOK. Azure OpenAI reachable; deployment={gen.azure_deployment}")
    return 0


def main() -> int:
    cfg = load_config()
    try:
        gen = Generator(cfg)
    except ValueError as exc:
        print(f"CONFIG. {exc}")
        return 1

    if gen.provider == "azure_openai":
        return _check_azure(gen)
    if gen.provider == "ollama":
        return _check_ollama(gen)

    print(f"Unknown MODEL_PROVIDER={gen.provider!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
