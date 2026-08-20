# Azure / AI Foundry deployment (Heinzy)

Run Heinzy with **Azure AI Foundry** for answers. No local Ollama or GPU required.

Retrieval (handbook + embeddings) still runs on your machine or VM. Only **generation** calls Azure.

---

## What you need

- Python 3.11+
- This repo (branch with Azure support)
- A handbook PDF
- An Azure AI Foundry (or Azure OpenAI) chat deployment and its **API key**

---

## 1. Install

```bash
cd AIGovernance
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -e ".[store,governance]"
```

---

## 2. Configure `.env`

Copy the template and edit:

```bash
cp .env.template .env
```

Set:

```
MODEL_PROVIDER=azure_openai

# Foundry-style host (no /openai/v1/... path required — code adds it)
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.services.ai.azure.com

# Exact model / deployment name from Foundry (e.g. Llama-3.3-70B-Instruct)
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME

# Key from THE SAME resource as the endpoint (Keys blade)
AZURE_OPENAI_API_KEY=YOUR_KEY

AZURE_OPENAI_API_VERSION=2024-10-21

CHROMA_HOST=127.0.0.1
CHROMA_PORT=8000
```

**Classic Azure OpenAI** (`https://YOUR_RESOURCE.openai.azure.com`) also works with the same variables; the client picks the right URL shape.

Never commit `.env`.

---

## 3. Handbook + vector store

Put the PDF here:

```text
data/corpus/your-handbook.pdf
```

**Option A — no Docker (simple):** in `config.yaml`:

```yaml
vector_store:
  backend: "memory"
```

Index is rebuilt each process start.

**Option B — persistent (Chroma):**

```yaml
vector_store:
  backend: "chroma"
```

```bash
docker compose -f docker-compose.chroma.yml up -d
```

---

## 4. Verify Azure, then ask

```bash
python scripts/check_model_host.py
# expect: style foundry_v1 (or classic models list) and OK

python scripts/ask_handbook.py --query "What does the MISM curriculum cover?"

python scripts/chat.py
```

---

## Common failures

| Symptom | Likely cause | Fix |
|--------|----------------|-----|
| Still talking to `11434` | `MODEL_PROVIDER` not set | Set `MODEL_PROVIDER=azure_openai` |
| `401` Access denied | Wrong key, or key for a different resource | Use key from the **same** resource as `AZURE_OPENAI_ENDPOINT` |
| `401` after pasting key | Key pasted twice (length ~160+) | Paste once; length is usually ~80–90 |
| `DeploymentNotFound` / 404 | Name is the resource/create id, not the model deployment | Use the deployment/model name from Foundry (e.g. `Llama-3.3-70B-Instruct`) |
| `no_retrieved_context` | Question didn’t clear the score floor (or wrong PDF) | Try a clearly in-handbook question first; confirm PDF is in `data/corpus/` |
| Edited `.env` but app ignores it | Different clone/user home on a shared VM | Run from the same directory that owns the `.env` you edited |

---

## GCP VM notes (optional)

- CPU-only VM is fine with Azure generation (do **not** run large Ollama models there).
- Prefer one Linux user and one repo path.
- Chroma + Python venv is enough; building the full app Docker image needs more disk.

Stop when idle:

```bash
docker compose -f docker-compose.chroma.yml down
# then stop the VM in GCP if you use one
```

---

## What stays local vs Azure

| Piece | Where |
|--------|--------|
| PDF, chunking, embeddings, Chroma/memory | Your machine / VM |
| Answer generation | Azure AI Foundry |
| Governance tool loop | Ollama-oriented today; leave unset for Azure chat-only |

---

## Quick copy-paste `.env` (Foundry)

```
MODEL_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.services.ai.azure.com
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8000
```
