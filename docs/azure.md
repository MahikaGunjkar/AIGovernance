# Azure AI Foundry / OpenAI — run Heinzy

Generation on **Azure**; app + handbook on your **laptop or VM**.  
No local GPU required for answers. Local Ollama path: [`README.md`](../README.md).

Handbook RAG, Layer 1/2 abstention, citations, and **governed tool calls**
(when `GOVERNANCE_*` is set) all work with `MODEL_PROVIDER=azure_openai`.
The deployment must support OpenAI-style function calling.

Most unit tests force Ollama so a local Azure `.env` does not break stubs;
Azure-specific coverage is in `tests/test_azure_generation.py`. Verify a live
host with `python scripts/check_model_host.py` and a real ask.

---

## Setup

Follow [README → How to run](../README.md#how-to-run) for clone, governance
worktree, venv, and `pip install -e ".[store,dev,governance,webui]"`. Then set `.env`:

```
MODEL_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.services.ai.azure.com
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
CHROMA_HOST=127.0.0.1

GOVERNANCE_SRC=../AIGovernance-governance/src
GOVERNANCE_POLICY_PATH=../AIGovernance-governance/policies/governance_policy.yaml
GOVERNANCE_HOST_PATH=../AIGovernance-governance
```

Key must come from the **same** Azure resource as the endpoint. Never commit `.env`.
Classic host also works: `https://YOUR_RESOURCE.openai.azure.com`.

**Store:** `config.yaml` → `vector_store.backend: memory`, or `chroma` +  
`docker compose -f docker-compose.chroma.yml up -d`.

On a VM prefer **chroma + Azure** (not large Ollama on CPU). Use one Linux user /
one checkout so `.env` edits match the process you run.

---

## Run

```bash
python scripts/check_model_host.py
python scripts/ask_handbook.py --query "What does the MISM curriculum cover?"
python scripts/chat.py
```

**Web UI** (http://localhost:5000):

```bash
pip install -e ".[webui]"
python -m heinzy.webui.app
```

Same `.env` as the CLI. Docker UI also passes Azure vars from `.env`:

```bash
docker compose -f docker-compose.webui.yml up --build
```

On a VM, tunnel if needed:  
`ssh -L 5000:127.0.0.1:5000 user@vm` then open http://localhost:5000 locally.

---

## Governance tools on Azure

With the governance worktree mounted, `Generator` enters the same tool loop used
for Ollama: OpenAI-format `tools` from `heinzy/tools/registry.py`, every call
gated by `OllamaGovernanceInterceptor` before execution.

| Capability | Azure |
|------------|--------|
| Handbook RAG + grounded answers | Yes |
| Layer 1 / Layer 2 abstention + citations | Yes |
| Governed `web_search` / write tools / approval pauses | Yes (function-calling deployment) |

If tools never fire: confirm `GOVERNANCE_SRC` / `GOVERNANCE_POLICY_PATH`, that
`governance.enabled` is true in `config.yaml`, and that the Azure deployment
supports function calling (not every Foundry model does).

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| Still on `:11434` | Set `MODEL_PROVIDER=azure_openai` |
| `401` | Wrong key / wrong resource; or key pasted twice (~160 chars) |
| `DeploymentNotFound` | Use the **model deployment** name, not the resource create id |
| `no_retrieved_context` | In-handbook question; try including **MISM** |
| Governance tools never fire | Worktree + `GOVERNANCE_*`; deployment must support function calling |
| Layer 1 `agt` / worktree issues | Confirm worktree + `GOVERNANCE_SRC` / `GOVERNANCE_POLICY_PATH` |
