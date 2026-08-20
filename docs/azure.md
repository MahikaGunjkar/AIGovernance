# Azure AI Foundry / OpenAI — run Heinzy

Generation on **Azure**; app + handbook on your **laptop or VM**.  
No local GPU required for answers. Local Ollama path: [`README.md`](../README.md).

Layer 1 abstention (`builtin` / `agt`) and citations work the same as Ollama.
**Governed tool calls do not:** Azure is chat-only until tool calling is wired
(see [Governance tools on Azure](#governance-tools-on-azure) below).

Verify with `python scripts/check_model_host.py` and a real ask — unit tests
force `MODEL_PROVIDER=ollama` and do not exercise Azure.

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

What already works with `MODEL_PROVIDER=azure_openai`:

- Handbook RAG + grounded answers
- Layer 1 refusal (score floor / policy engine) and Layer 2 sentinel abstention
- Citation checks

What does **not** yet:

- The Ollama tool-calling loop gated by `OllamaGovernanceInterceptor`
  (`web_search`, write tools, human-approval pauses)

`Generator.use_tools` stays off for Azure on purpose. Setting `GOVERNANCE_*`
still mounts policy for Layer 1 `agt` if you use that engine; it does not
enable tool calls. For governed tools today, use `MODEL_PROVIDER=ollama`.

Wiring tools on Azure is moderate work (not a config flip): pass OpenAI-format
`tools` (already in `heinzy/tools/registry.py`), map Azure `tool_calls` /
`role=tool` messages into the existing loop in `_chat_azure`, then allow
`use_tools` when `governance_available()`. Most models on Foundry that support
function calling can use the same path.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| Still on `:11434` | Set `MODEL_PROVIDER=azure_openai` |
| `401` | Wrong key / wrong resource; or key pasted twice (~160 chars) |
| `DeploymentNotFound` | Use the **model deployment** name, not the resource create id |
| `no_retrieved_context` | In-handbook question; try including **MISM** |
| Governance tools never fire | Expected on Azure today — use Ollama, or wait for tool wiring above |
| Layer 1 `agt` / worktree issues | Confirm worktree + `GOVERNANCE_SRC` / `GOVERNANCE_POLICY_PATH` |
