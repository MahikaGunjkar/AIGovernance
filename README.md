# Heinzy — Governed Advising Assistant

A governed internal assistant for Heinz College advisors. It answers program and
policy questions from official MISM handbook documents, returns verifiable
citations, and runs inside a governance layer that constrains what it can access
and do.

## Status

- ✅ Ingestion — PDF → chunks → embeddings
- ✅ Retrieval — Chroma-backed
- ✅ Generation — Gemma via Ollama
- ✅ Grounded answering + abstention (A3b) — answers only from retrieved chunks,
  refuses 10/10 out-of-corpus questions on the real handbook
  ([how it works](#grounded-answering--abstention-a3b))
- ✅ Docker image — builds and runs standalone; real answers need Gemma + Chroma reachable (see below)
- 🟡 Event log — retrieval only, generation not logged yet
- ⬜ Eval harness
- ⬜ Governance layer

Details: [Ownership & what's done](#ownership--whats-done).

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

### Embeddings

Both ingest and retrieval embed with `fastembed` (`BAAI/bge-small-en-v1.5`) —
core dependency now, ~130MB, CPU-only, no API key. Same library both sides, so
vectors are always comparable. Falls back to a hash embedding
(`HASH-FALLBACK` in output) only if `fastembed` fails to import.

### Docker — Heinzy app image (S2)

```bash
docker build -t heinzy .
docker run --rm heinzy          # runs the retrieval smoke test
```

Build installs the `store` extra and pre-warms the `fastembed` cache, so the
image is fully self-contained. No Gemma weights, no Ollama — generation calls
the shared host.

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
- Align `config.yaml` → `model.tag` with what you pulled, or set `MODEL_TAG`
  to override it locally for testing without touching the shared config.
- Generation (`heinzy/generation/generator.py`) calls `/api/chat` on this
  endpoint — see [Ask a real question](#ask-a-real-question-end-to-end).

### Shared Chroma host (S4) — one machine for the whole team

Same pattern as Gemma: **one shared Chroma Docker service**, Mac/Windows clients
only set `CHROMA_HOST`. Default backend is now `chroma`; flip to `memory` in
`config.yaml` for offline work (tests use their own store either way).

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

### Ask a real question end to end

With Gemma (`MODEL_ENDPOINT`) and Chroma (`CHROMA_HOST`) reachable:

```bash
# PDF goes in data/corpus/ first (gitignored, shared out of band)
python scripts/ask_handbook.py --query "how many electives can I take?"
python scripts/chat.py   # interactive
```

First run ingests; later runs against the same doc skip straight to
retrieval + generation (`store.has_doc()` check).

---

## Grounded answering & abstention (A3b)

The assistant answers **only** from retrieved handbook chunks, and when the
handbook doesn't contain the answer it says so instead of producing plausible
text. An advisor repeating an invented policy to a student is the failure this
is built to prevent, so "no answer" is treated as strictly better than a
confident guess.

### Two layers, because either alone leaks

| Layer | Fires when | Mechanism | `refusal_reason` |
|-------|-----------|-----------|------------------|
| 1, no context | fewer than `min_hits` chunks clear `retrieval.score_floor` | **the model is never called**, so it cannot be talked out of it | `no_retrieved_context` |
| 2, insufficient context | chunks cleared the floor but don't answer the question | model emits `generation.abstain.sentinel`, which we detect and convert to a refusal | `model_insufficient_context` |

Layer 2 is not optional. Nearest neighbour search always returns *k* chunks and
has no way to report "no match", so every question retrieves something that
scores respectably. Measured on the real handbook with
`scripts/calibrate_floor.py`.

```
lowest in-corpus top score  : 0.7698   (ic-06, capstone project)
highest out-of-corpus score : 0.7449   (ooc-07, an invented transfer policy)
Separable. Suggested retrieval.score_floor: 0.757  (margin 0.0249)
```

Separable, but by 0.0249 across 16 questions. A floor set at that razor edge
would refuse the first legitimate question that happens to score 0.74, and over
refusal is the failure mode that makes the assistant useless rather than merely
unhelpful. So `score_floor` stays conservative. At 0.65 the gate drops 3 of 10
out-of-corpus questions for free while keeping all 6 controls with 0.12 of
headroom, and layer 2 catches the rest.

The suggested floor is not a number to paste into config. Re-run the calibration
after any change to chunking, `k`, or the embedding model, and treat a thin
margin as evidence that the floor should stay loose.

Refusals return the configured `refusal_text`, never model prose, so the eval
harness, event log, and UI branch on `Answer.refused` instead of pattern
matching English.

### Citations are checked, not trusted

Every section an answer cites is matched against the sections retrieval actually
returned, and mismatches land in `Answer.unsupported_citations` and fail the
eval. This is a *necessary* condition for grounding rather than a sufficient
one. It proves nothing outside the retrieved set was cited, not that every
sentence is supported. Claim level faithfulness scoring belongs to the eval
harness (A6).

### Running the eval

```bash
# No handbook PDF needed. Labelled stand-in corpus, in-memory, no Chroma.
python scripts/eval_abstention.py --fixture

# Against the real handbook in data/corpus/
python scripts/eval_abstention.py

# Same, when the shared Chroma host is unreachable
python scripts/eval_abstention.py --backend memory

# Whatever model is actually pulled locally
MODEL_TAG=llama3.2:latest python scripts/eval_abstention.py --fixture

# Is a score floor viable on this corpus? Retrieval only, no model calls.
python scripts/calibrate_floor.py
```

Exits non-zero if any out-of-corpus question got answered, any in-corpus control
got refused, or any answer cited a section retrieval never returned, so it can
gate a PR. Per-question results are written to `data/logs/`, which is gitignored.

**The in-corpus controls are part of the pass criteria on purpose.** A system
that refuses everything scores a perfect 10/10 on the out-of-corpus set and is
useless, and measuring only one direction hides that.

### Results

`eval/abstention_questions.yaml` holds 10 out-of-corpus questions. Each targets
a different way a RAG system invents an answer, covering world knowledge, an
adjacent program, the wrong institution,
personal advice, absent specifics, a fact that lives in another document, a
question presupposing a policy that doesn't exist, an out-of-scope task, and two
topics that sound like guaranteed handbook sections but are absent from this
one. Six in-corpus controls sit alongside them.

Every out-of-corpus question's absence was **verified against the handbook**
rather than assumed, and every control was verified present. Both directions
matter, because an unanswerable control fails for the wrong reason and makes a
correct refusal look like a bug.

Real handbook (`mism-student-handbook.pdf`, 14 pages into 28 chunks),
`config_hash=7502dd99ab55`, k=5, temperature 0.

| Model | Out-of-corpus refused | Controls answered | Unsupported citations | Contentless answers |
|-------|----------------------|-------------------|-----------------------|---------------------|
| `gemma4:e2b` | **10/10** | 6/6 | 0 | 0 |
| `llama3.2:latest` | **9/10** | 6/6 | 0 | 0 |
| `gemma2:9b` (team target) | *not run, needs the shared host* | | | |

The single `llama3.2` miss is `ooc-04`, where it deflects to Career Services
rather than inventing a policy. Not a refusal, but not the fabrication the
criterion targets either. `gemma4:e2b` refuses it outright.

Two reported but non-failing signals guard against passing on a technicality.

- **contentless answers**, where a bare citation with no prose is not an answer
  but is not a refusal either, so it would otherwise score as a success.
- **answers citing nothing**, where the answer may be perfectly grounded but
  with no citation to check, the grounding check passed on an empty set. A clean
  run is not the same as a verified one.

### Reproducibility

Runs are byte identical, with **16 of 16 answers matching across two consecutive
runs** of the same model. That required pinning `generation.temperature` and
`generation.seed`. Ollama defaults to temperature 0.8, which had the same
question refusing on one run and answering on the next. A governed assistant
that answers differently on a re-ask cannot be audited, and the `config_hash`
stamped into every result was promising a reproducibility the system did not
have.

Because prompt driven abstention is model dependent, every report stamps
`model_tag`, `embed_model`, `k`, `score_floor`, `temperature`, `seed` and
`config_hash`. A pass is evidence for that combination rather than a permanent
property, so re-run before trusting it on a different model.

### Known issues this surfaced elsewhere

Neither is an A3b bug, but both degrade grounded answering and belong to other
areas.

- **Ingestion (A1).** The page footer `MISM Handbook Addendum <n>` is extracted
  into chunk text on nearly every page. It pollutes context and invites the
  model to cite a footer as if it were a section.
- **Retrieval (A2).** "What concentrations can a MISM student pursue?" is
  answered by the handbook but not by top-5 retrieval. Section 7 says only that
  concentrations need no extra electives, and the five names live in 7.1 through
  7.5, none of which surface for that phrasing. A refusal there is correct given
  the context, so it was dropped as a control rather than counted as over
  refusal.

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
  ingest/                # ingestion pipeline, M0–M6 (task A1) — done
    registry.py          #   M0 hash PDFs -> doc_id
    extract.py           #   M1 PDF -> per-page text
    structure.py         #   M2 pages -> section-aware blocks
    chunk.py             #   M3 blocks -> chunks
    embed.py             #   M4 chunks -> vectors
    index.py             #   M5 vectors -> collection
    verify.py            #   M6 sanity checks
    types.py             #   shared record types + pre/post contracts
  retrieval/             # retrieval layer (task A2) — done
    retrieve.py          #   question -> top-k chunks, k from config
    embedder.py          #   fastembed (same as ingest's, for compatible vectors)
    store.py             #   VectorStore protocol + in-memory/chroma adapters
    stores/chroma_store.py  #   Chroma HTTP adapter
  generation/             # generation layer (task A3) — done
    generator.py          #   chunks -> grounded, cited answer via Ollama
    grounding.py          #   citation extraction + retrieved-set check (A3b)
  eval/
    abstention.py         #   out-of-corpus refusal eval + fixture store (A3b)
  pipeline.py             # shared ingest-and-populate-store orchestration
eval/
  abstention_questions.yaml  # 8 out-of-corpus + 5 in-corpus control questions
  fixture_corpus.yaml        # labelled stand-in corpus (NOT the real handbook)
scripts/
  ask_handbook.py         # ingest + retrieve + generate, one question via --query
  chat.py                 # same, interactive
  eval_abstention.py      # proves the A3b claim; non-zero exit on failure
  calibrate_floor.py      # is a score floor viable on this corpus?
  smoke_retrieval.py      # placeholder-chunk demo, no real data needed
  smoke_store.py          # store factory smoke (memory or chroma)
tests/test_retrieval.py      # locks the retrieval contract
tests/test_store.py          # locks S4 get_store / adapter contract
tests/test_generation_grounding.py  # locks the refusal + citation contract (A3b)
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
| Ingestion (A1) | ✅ done | M0–M6 implemented, `verify()` clean on the real handbook |
| Retrieval (A2) | ✅ done | config-driven `k`, provenance on every hit, tests green |
| Vector-store seam (S4) | ✅ memory + chroma | `InMemoryStore` default; shared Chroma via `docker-compose.chroma.yml` + `ChromaStore` HttpClient |
| Docker (S2) | ✅ done | pinned base + deps !
| Shared Gemma host (S3) | 🟡 compose ready | `docker-compose.ollama.yml` — one host for the team; A3 still must call it |
| Ingestion bodies (A1) | ⬜ skeleton only | functions `raise NotImplementedError`; contracts written in docstrings |
| Generation/answering (A3) | ✅ done | calls shared `MODEL_ENDPOINT` (Gemma via Ollama) |
| Grounded answering + abstention (A3b) | ✅ done | two-layer refusal, citation check, eval harness. On the real handbook with `gemma4:e2b`, 10/10 out-of-corpus refused, 6/6 controls answered, 0 unsupported citations. Still to run on `gemma2:9b` |
| Citations (A4) | 🟡 checked, not rendered | cited sections verified against the retrieved set; provenance flows from retrieval |
| Event log (A5) | ⬜ not started | append-only audit log (out of scope on this branch) |
| Eval harness (A6) | ⬜ not started | owner: Lisa |
| Governance layer | ⬜ not started | scaffolding on `feature/governance-policies`, not wired in — `agent_governance` dep doesn't exist |

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
