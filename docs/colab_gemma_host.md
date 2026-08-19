# Colab Gemma host runbook (S3 / issue #16)

Shared inference for Heinzy: **Gemma 3 12B via Ollama on Google Colab**, exposed
with **ngrok**. App code is unchanged — set `MODEL_ENDPOINT` to the tunnel URL.

Canonical notebook: [`notebooks/colab_gemma_ollama_host.ipynb`](../notebooks/colab_gemma_ollama_host.ipynb)

## Ownership

| Role | Who | Duty |
|------|-----|------|
| Host operator | **@asriram15** | Start/restart Colab, pull model, post new `MODEL_ENDPOINT` when it changes |
| Everyone else | Team | Put current URL in `.env`; ping operator if generation dies |

Do **not** commit live tunnel URLs. Record them in team chat and local
gitignored `TEAM_INFRA_NOTES.md`.

## Operator checklist

1. Open the notebook in Google Colab (File → Upload, or open from the repo).
2. Runtime → Change runtime type → **GPU**.
3. Colab Secrets → add `NGROK_AUTHTOKEN` (ngrok dashboard → Your Authtoken) → enable **Notebook access**.
4. Runtime → Run all.
5. Confirm the self-check cell prints `gemma3:12b` in `/api/tags` and a chat reply.
6. Copy `MODEL_ENDPOINT=https://...` into **team chat** + `TEAM_INFRA_NOTES.md`.
7. Leave the tab/runtime awake while the team needs generation.

**Note:** Do not use `curl … install.sh | sh` on Colab — it often fails with
`CalledProcessError` (systemd/service setup). The notebook installs the official
`ollama-linux-amd64.tgz` binary instead.

### After disconnect / restart

Colab and ngrok URLs are ephemeral. Re-run the notebook and **post the new
URL again**. Teammates with a stale `.env` will see connection errors on
`/api/chat` with no other warning.

## Teammate checklist

```bash
# .env — use the URL from team chat (HTTPS, no :11434 suffix required)
MODEL_ENDPOINT=https://REPLACE_WITH_CURRENT_NGROK_HOST

# Sanity (should list gemma3:12b):
curl -sS "$MODEL_ENDPOINT/api/tags"
```

Align `config.yaml` → `model.tag: gemma3:12b` (repo default after #16), or
override for one shell:

```bash
MODEL_TAG=gemma3:12b python scripts/ask_handbook.py --query "how many electives?"
```

## Optional: local LAN Ollama

[`docker-compose.ollama.yml`](../docker-compose.ollama.yml) remains for a
persistent GPU box / laptop demos. That path still uses
`MODEL_ENDPOINT=http://<host>:11434`. Prefer Colab for the shared team host.

## Security notes

- ngrok exposes Ollama on the public internet; protect the ngrok account and
  rotate tokens if leaked.
- Do not put authtokens in the notebook source or commit them.
