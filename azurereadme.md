# Azure AI Foundry / OpenAI — run Heinzy

Generation on **Azure**; handbook retrieval on your **laptop or VM**.  
No Google Colab. No local GPU required for answers.

Also see the top of [`README.md`](README.md) for the local-Ollama path.

---

## Two setups

### A) Local laptop + Azure answers

App, PDF, and embeddings run on your machine. Only chat completions hit Azure.

### B) VM + Azure answers

Same as A, but the app/Chroma run on a cloud VM (e.g. GCP) so your laptop stays light. Generation still uses Azure — do not run large Ollama models on a small CPU VM.

---

## Prerequisites

- Python 3.11+
- This repo
- Handbook PDF
- Azure AI Foundry **or** Azure OpenAI:
  - resource endpoint
  - **deployment / model name** (e.g. `Llama-3.3-70B-Instruct`)
  - API key from **that same** resource

---

## Install (laptop or VM)

```bash
cd AIGovernance
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e ".[store,dev,governance]"
cp .env.template .env
mkdir -p data/corpus
# copy your handbook PDF into data/corpus/
```

---

## Configure `.env` for Azure

```
MODEL_PROVIDER=azure_openai

# Foundry (AI Services) — host only; code appends /openai/v1/chat/completions
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.services.ai.azure.com

# Exact deployment/model name from the portal
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME

# Key from the SAME resource (Keys blade)
AZURE_OPENAI_API_KEY=YOUR_KEY

AZURE_OPENAI_API_VERSION=2024-10-21

CHROMA_HOST=127.0.0.1
CHROMA_PORT=8000
```

Classic Azure OpenAI (`https://YOUR_RESOURCE.openai.azure.com`) uses the same
variables; the client picks the URL shape automatically.

Never commit `.env`.

---

## Vector store

**No Docker (simplest):** in `config.yaml`:

```yaml
vector_store:
  backend: "memory"
```

**Persistent (recommended on a VM):**

```yaml
vector_store:
  backend: "chroma"
```

```bash
docker compose -f docker-compose.chroma.yml up -d
curl http://127.0.0.1:8000/api/v2/heartbeat
```

---

## Run the bot

```bash
python scripts/check_model_host.py
# expect: foundry_v1 (or classic) and OK

python scripts/ask_handbook.py --query "What does the MISM curriculum cover?"

python scripts/chat.py
```

Optional UI:

```bash
pip install -e ".[webui]"
python -m heinzy.webui.app
# http://localhost:5000
```

Tip: include **MISM** in questions when retrieval feels weak — embeddings match
handbook wording.

---

## VM hosting checklist (setup B)

1. SSH to the VM; clone repo; create venv; `pip install -e ".[store,governance]"`.
2. Copy PDF → `data/corpus/`.
3. Start Chroma; set `backend: chroma` and `.env` as above.
4. Run `check_model_host.py`, then `chat.py` / `ask_handbook.py`.
5. Prefer **one** Linux user and one checkout path (avoid editing `.env` in a
   different home than the one you run from).
6. Stop when idle: `docker compose -f docker-compose.chroma.yml down`, then stop
   the VM if you use cloud compute.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|--------|----------------|-----|
| Still using `11434` | `MODEL_PROVIDER` unset | Set `MODEL_PROVIDER=azure_openai` |
| `401` | Wrong key or wrong resource | Key must belong to the endpoint’s resource |
| `401` after paste | Key pasted twice (~160+ chars) | Paste once (~80–90 chars typical) |
| `DeploymentNotFound` / 404 | Resource/create name ≠ deployment | Use the **model deployment** name from Foundry |
| `no_retrieved_context` | Score floor / weak phrasing | Try a clearly in-handbook question; add “MISM” |
| Edited `.env` but ignored | Different clone/user on the VM | Run from the directory that owns that `.env` |

---

## What runs where

| Piece | Laptop / VM | Azure |
|--------|-------------|-------|
| PDF, chunking, embeddings, Chroma/memory | ✅ | |
| `ask` / `chat` / web UI | ✅ | |
| LLM answer generation | | ✅ |
| Governance tool loop | Ollama-oriented; leave off for Azure chat-only | |
