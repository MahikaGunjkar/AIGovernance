"""
Check that the configured generation host is reachable and serving our model.

The shared Gemma host is planned to run on Colab, which hands out a new tunnel
URL every time it restarts and drops the session after a few hours. The failure
that causes is quiet. MODEL_ENDPOINT keeps pointing at a dead URL, and the first
sign of trouble is a timeout in the middle of someone's eval run, or worse, a
run that reaches a host serving a different model than the one config.yaml
claims and stamps the wrong tag onto the results.

pre:  config.yaml is readable, MODEL_ENDPOINT is set in .env or the environment
post: exits 0 when the host answers and carries config.model.tag, non-zero
      otherwise, so it can gate a run or sit in CI
invariant: reads the endpoint and the tag from config, never hardcoded, and
           reports the endpoint it actually used so a stale URL is visible
           rather than guessed at

Run from repo root:
    python scripts/check_model_host.py
    MODEL_ENDPOINT=https://abc123.ngrok.app python scripts/check_model_host.py
"""
from __future__ import annotations

import sys

import requests

from heinzy.common.config import load_config
from heinzy.generation.generator import Generator


def main() -> int:
    cfg = load_config()
    gen = Generator(cfg)
    print(f"endpoint  {gen.endpoint}")
    print(f"wanted    {gen.model_tag}")
    print(f"auth      {'basic auth from MODEL_BASIC_AUTH' if gen.auth else 'none'}")

    try:
        resp = requests.get(f"{gen.endpoint}/api/tags", timeout=10, auth=gen.auth)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"\nUNREACHABLE. {type(exc).__name__} talking to {gen.endpoint}")
        print("If the host runs on Colab the tunnel URL changes on every restart, "
              "so check the current URL with whoever owns the host and update "
              "MODEL_ENDPOINT in your .env.")
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


if __name__ == "__main__":
    sys.exit(main())
