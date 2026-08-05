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

### Docker

```bash
docker build -t heinzy .
docker run --rm heinzy          # runs the retrieval smoke test
```

The image runs the Python code only. The Gemma 12B model is served separately
(task S3); this container reaches it via `MODEL_ENDPOINT`.

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
scripts/smoke_retrieval.py   # end-to-end demo on placeholder chunks
tests/test_retrieval.py      # locks the retrieval contract
data/corpus/             # MISM PDFs go here (gitignored, shared out of band — S6)
data/index/              # built indexes (gitignored)
```

---

## Ownership & what's done

| Area | Status | Notes |
|------|--------|-------|
| Config system (S5) | ✅ done | `config.yaml` + loader + hash |
| Retrieval (A2) | ✅ done | config-driven `k`, provenance on every hit, tests green |
| Vector-store seam (S4) | ✅ interface done | in-memory now; **DB owner** implements the `VectorStore` protocol and adds a branch in `get_store()` — retrieval code untouched |
| Docker (S2) | ✅ done | pinned base + deps |
| Ingestion bodies (A1) | ⬜ skeleton only | functions `raise NotImplementedError`; contracts written in docstrings |
| Generation/answering (A3) | ⬜ not started | Gemma 12B, served separately (S3) |
| Citations (A4) | ⬜ not started | provenance already flows from retrieval |
| Event log (A5) | 🟡 shape defined | `RetrievalResult.to_log_record()` emits the A5 record shape |
| Eval harness (A6) | ⬜ not started | owner: Lisa |
| Governance layer | ⬜ not started | policy engine, HITL, red-team set |

### Swapping in the real database

The DB owner does **not** touch retrieval code. They:
1. Write a class implementing the `VectorStore` protocol in `heinzy/retrieval/store.py`.
2. Register it in `get_store()` (one `if` branch).
3. Set `vector_store.backend` in `config.yaml`.

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

Copy `.env.template` to `.env` and fill values. A clean clone runs after filling
the template (Infra task S7). `.env` is gitignored.
