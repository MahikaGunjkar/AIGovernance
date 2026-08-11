# Heinzy — Governed Advising Assistant

A governed internal assistant for Heinz College advisors. It answers program and
policy questions from official MISM handbook documents, returns verifiable
citations, and runs inside a governance layer that constrains what it can access
and do.

This repo currently covers the **ingestion pipeline skeleton** and a **working,
config-driven retrieval layer**. The generation model, the real vector DB, and
the governance layer are tracked as separate owners' tasks (see
[Ownership](#ownership--whats-done)).

---

## Quick start (clone → running)

```bash
git clone <this-repo> && cd heinzy
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e .

# Prove the retrieval pipeline works end to end (uses placeholder chunks):
python scripts/smoke_retrieval.py

# Change k without editing any source — it's read from config:
python scripts/smoke_retrieval.py --k 2 --query "internship waiver"

# Run the tests:
pip install -e ".[dev]"
pytest -q
```

That's the S1 "done when": a stranger clones and reaches a running system from
this README alone.

### Real embeddings (optional)

By default the embedder falls back to a **deterministic hash embedding** so the
pipeline runs with zero downloads — scores are *not* semantically meaningful and
the output clearly says `HASH-FALLBACK`. For real retrieval quality:

```bash
pip install -e ".[embed]"   # installs sentence-transformers, downloads BGE
```

The output then reports `semantic` and rankings become meaningful.

### Docker — Heinzy app image (S2)

```bash
docker build -t heinzy .
docker run --rm heinzy          # runs the retrieval smoke test
```

This image runs the Python retrieval/ingest code only. It does **not** bundle
Gemma weights.

### Shared Gemma host (S3) — one machine for the whole team

Pattern: **one shared Ollama host**, everyone else only sets `MODEL_ENDPOINT`.

| Role | What they do |
|------|----------------|
| **Host operator** (one person / one GPU box) | Runs [`docker-compose.ollama.yml`](docker-compose.ollama.yml), pulls the model, keeps port `11434` reachable on LAN/VPN |
| **Everyone else** | Copy `.env.template` → `.env`, set `MODEL_ENDPOINT=http://<host-ip-or-name>:11434` |

**Host operator setup**

```bash
# On the shared machine (Docker + NVIDIA GPU recommended):
docker compose -f docker-compose.ollama.yml up -d

# Pull once (cached in the named volume). Prefer 9b on 8GB VRAM; 12b if it fits:
docker compose -f docker-compose.ollama.yml exec ollama ollama pull gemma2:9b
# docker compose -f docker-compose.ollama.yml exec ollama ollama pull gemma2:12b
```

Tell teammates the reachable URL (e.g. `http://10.0.0.12:11434`). They put that
in `.env` — nothing else to install for the model.

**Teammate check** (from any machine that can reach the host):

```bash
curl http://<host>:11434/api/tags
```

**Notes**

- Keep the host on a private network / VPN. Ollama on `11434` has no app-level
  auth in this setup — do not expose it to the public internet.
- Open firewall TCP `11434` to teammates only.
- If the host has no GPU, remove the `gpus: all` line in the compose file;
  inference still works on CPU, just slowly.
- Align `config.yaml` → `model.tag` with the Ollama model you pulled
  (`gemma2:9b` / `gemma2:12b`) when A3 lands. Recorded tag is for reproducibility.
- Generation/answering (A3) is not implemented yet; this only stands up the
  shared endpoint the team will call.

### Shared Chroma host (S4) — one machine for the whole team

Same pattern as Gemma: **one shared Chroma Docker service**, Mac/Windows clients
only set `CHROMA_HOST`. Default config stays `backend: memory` so clone + pytest
work offline.

| Role | What they do |
|------|----------------|
| **Host operator** | Runs [`docker-compose.chroma.yml`](docker-compose.chroma.yml), opens firewall TCP **8000** (Private), keeps it up |
| **Everyone else** | `pip install -e ".[store]"` (thin `chromadb-client`, Mac/Windows), set `CHROMA_HOST=<host-ip>` in `.env`, set `vector_store.backend: chroma` |

**Host operator setup** (Windows or Mac Docker Desktop — CPU only):

```bash
docker compose -f docker-compose.chroma.yml up -d
curl http://127.0.0.1:8000/api/v2/heartbeat   # or /api/v1/heartbeat on older images
```

**Teammate / host client check:**

```bash
pip install -e ".[store]"
# .env: CHROMA_HOST=127.0.0.1   (host) or CHROMA_HOST=<lan-ip> (teammates)
# config.yaml: vector_store.backend: chroma
python scripts/smoke_store.py
```

**Notes**

- Do not expose `:8000` to the public internet (no app-level auth here).
- Retrieval code never imports Chroma — only `heinzy/retrieval/stores/chroma_store.py`.
- Flip back to `backend: memory` anytime for offline work.

---

## Configuration — one file, no constants in source

All tunable behavior lives in [`config.yaml`](config.yaml): chunk size, overlap,
`k`, embedding model, vector-store backend, generation model tag/quantization,
prompt version. **Changing retrieval behavior means editing config, not code.**

The sha256 of the config is exposed as `config_hash` and stamped into every
retrieval log record and (later) every eval result, so any run is reproducible.

```bash
python heinzy/common/config.py   # prints version, config_hash, key values
```

---

## Layout

```
config.yaml              # single source of truth for all knobs (S5)
heinzy/
  common/config.py       # config loader + config_hash (S5)
  ingest/                # ingestion pipeline skeleton, M0–M6 (task A1, owner: Guy)
    registry.py          #   M0 hash PDFs -> deterministic doc_id
    extract.py           #   M1 PDF -> per-page text
    structure.py         #   M2 pages -> section-aware blocks
    chunk.py             #   M3 blocks -> chunks
    embed.py             #   M4 chunks -> vectors
    index.py             #   M5 vectors -> collection
    verify.py            #   M6 read-only sanity checks
    types.py             #   shared record types + pre/post contracts
  retrieval/             # retrieval layer (task A2) — THIS IS BUILT
    retrieve.py          #   question -> top-k chunks, k from config
    embedder.py          #   local BGE, hash fallback
    store.py             #   VectorStore protocol + in-memory adapter (S4 seam)
    stores/chroma_store.py  #   shared Chroma HTTP adapter (S4)
scripts/smoke_retrieval.py   # end-to-end demo on placeholder chunks
scripts/smoke_store.py       # S4 store factory smoke (memory or chroma)
tests/test_retrieval.py      # locks the retrieval contract
tests/test_store.py          # locks S4 get_store / adapter contract
docker-compose.ollama.yml    # shared Gemma/Ollama host for the team (S3)
docker-compose.chroma.yml    # shared Chroma host for the team (S4)
data/corpus/             # MISM PDFs go here (gitignored, shared out of band — S6)
data/index/              # built indexes (gitignored)
```

---

## Ownership & what's done

| Area | Status | Notes |
|------|--------|-------|
| Config system (S5) | ✅ done | `config.yaml` + loader + hash |
| Retrieval (A2) | ✅ done | config-driven `k`, provenance on every hit, tests green |
| Vector-store seam (S4) | ✅ memory + chroma | `InMemoryStore` default; shared Chroma via `docker-compose.chroma.yml` + `ChromaStore` HttpClient |
| Docker (S2) | ✅ done | pinned base + deps |
| Shared Gemma host (S3) | 🟡 compose ready | `docker-compose.ollama.yml` — one host for the team; A3 still must call it |
| Ingestion bodies (A1) | ⬜ skeleton only | functions `raise NotImplementedError`; contracts written in docstrings |
| Generation/answering (A3) | ⬜ not started | calls shared `MODEL_ENDPOINT` (Gemma via Ollama) |
| Citations (A4) | ⬜ not started | provenance already flows from retrieval |
| Event log (A5) | ⬜ not started | append-only audit log (out of scope on this branch) |
| Eval harness (A6) | ⬜ not started | owner: Lisa |
| Governance layer | ⬜ not started | policy engine, HITL, red-team set |

### Swapping vector-store backends

Retrieval talks only to the `VectorStore` protocol. To use the shared Chroma host:
1. Host runs `docker-compose.chroma.yml`.
2. Clients `pip install -e ".[store]"` and set `CHROMA_HOST` in `.env`.
3. Set `vector_store.backend: chroma` in `config.yaml` (no retrieval code changes).

To add another DB later: implement the protocol, register one branch in
`get_store()`, point `backend` at it.

### Note on infrastructure

We are using **Git only** — CMU Box / SVN is not required for this project.
Shared artifacts (corpus PDFs, built indexes) are exchanged out of band and kept
out of the repo (see `.gitignore`), per infra task S6.

---

## Branch & review convention

- `main` is protected; no direct pushes.
- Branch naming: `feature/<area>-<short-desc>` (e.g. `feature/retrieval-scorefloor`),
  `fix/<short-desc>`.
- Open a PR into `main`; at least one teammate review before merge.
- Keep tunables in `config.yaml`, never as literals in source (S5).
- Nothing secret is committed — use `.env` (copy from `.env.template`).

---

## Environment

Copy `.env.template` to `.env` and fill values. For retrieval smoke/tests you can
leave a placeholder `MODEL_ENDPOINT` — retrieval does not call it. For generation
(A3), set it to the **shared** Gemma host URL the team agreed on (Infra S7).
`.env` is gitignored.

Also update the local `.env` if you already had `localhost` from an older
template.
