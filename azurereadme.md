# Azure AI Foundry / OpenAI — run Heinzy

Generation on **Azure**; app + handbook on your **laptop or VM**.  
No Colab. No local GPU required for answers. Local Ollama path: [`README.md`](README.md).

---

## Install (same as README, with governance)

```bash
git clone <this-repo> && cd AIGovernance
git worktree add ../AIGovernance-governance feature/governance-policies

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -e ".[store,dev,governance]"
cp .env.template .env
mkdir -p data/corpus   # put handbook PDF here
```

`.env` (Azure + governance):

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

**Store:** `config.yaml` → `backend: memory`, or `chroma` +  
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

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| Still on `:11434` | Set `MODEL_PROVIDER=azure_openai` |
| `401` | Wrong key / wrong resource; or key pasted twice (~160 chars) |
| `DeploymentNotFound` | Use the **model deployment** name, not the resource create id |
| `no_retrieved_context` | In-handbook question; try including **MISM** |
| Governance tools never fire | Confirm worktree + `GOVERNANCE_SRC` / `GOVERNANCE_POLICY_PATH` |
