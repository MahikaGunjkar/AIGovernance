# Heinzy — Governed Advising Assistant

A governed internal assistant for Heinz College advisors. It answers program and
policy questions from official MISM handbook documents, returns verifiable
citations, and can run inside a governance layer that constrains what it can
access and do.

## Status

- ✅ Ingestion — PDF → chunks → embeddings
- ✅ Retrieval — Chroma or in-memory store
- ✅ Generation — **Azure AI Foundry / Azure OpenAI**, or local Ollama
- ✅ Grounded answering + abstention (A3) — answers only from retrieved chunks;
  refuses when the handbook does not support an answer
  ([how it works](#grounded-answering--abstention-a3b))
- ✅ Docker image — builds and runs standalone; generation needs a model endpoint
- 🟡 Event log — retrieval only, generation not logged yet
- ⬜ Eval harness
- ⬜ Governance layer (PEP lives on `feature/governance-policies`; this branch mounts it in Docker)

Details: [Ownership & what's done](#ownership--whats-done).

---

## How to run

Two supported ways to get answers:

| Mode | Where the app runs | Where the LLM runs | Docs |
|------|--------------------|--------------------|------|
| **Local** | Your laptop | Local Ollama **or** Azure | below |
| **Azure generation** | Laptop or VM | Azure AI Foundry / OpenAI only | below + [`azurereadme.md`](azurereadme.md) |

Retrieval (PDF → embeddings → Chroma/memory) always runs with the app. Only
**generation** calls the model host.

### 1. Clone and install

```bash
git clone <this-repo> && cd AIGovernance   # or your checkout name
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e ".[store,dev,governance]"
cp .env.template .env
```

Put the handbook PDF in `data/corpus/` (create the folder if needed).

### 2a. Local — Ollama on this machine

1. Install and start [Ollama](https://ollama.com/), then:

```bash
ollama pull gemma3:12b    # or a smaller tag for CPU-only machines
```

2. In `.env`:

```
MODEL_PROVIDER=ollama
MODEL_ENDPOINT=http://localhost:11434
```

3. In `config.yaml`, for a no-Docker trial:

```yaml
vector_store:
  backend: "memory"
```

Or use Chroma (persistent):

```bash
# config.yaml → vector_store.backend: chroma
# .env → CHROMA_HOST=127.0.0.1
docker compose -f docker-compose.chroma.yml up -d
```

4. Run:

```bash
python scripts/check_model_host.py
python scripts/ask_handbook.py --query "What does the MISM curriculum cover?"
python scripts/chat.py
```

Optional GPU/LAN Ollama box: [`docker-compose.ollama.yml`](docker-compose.ollama.yml)
and set `MODEL_ENDPOINT=http://<that-host>:11434`.

### 2b. Local app + Azure for generation (recommended for CPU-only)

App and handbook stay on your machine; answers come from Azure AI Foundry.

1. Create a Foundry / Azure OpenAI chat deployment and copy endpoint, deployment
   name, and **API key from that same resource**.
2. In `.env`:

```
MODEL_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.services.ai.azure.com
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
CHROMA_HOST=127.0.0.1
```

3. Choose `memory` or Chroma as in 2a.
4. Run:

```bash
python scripts/check_model_host.py
python scripts/ask_handbook.py --query "What does the MISM curriculum cover?"
python scripts/chat.py
```

Full Azure checklist and failure table: [`azurereadme.md`](azurereadme.md).

### 2c. Host the app on a VM (generation still Azure)

Use this when you want the laptop free of Docker/RAM load.

1. On a Linux VM (CPU is fine): install Docker + Python, clone the repo, copy the
   PDF to `data/corpus/`.
2. Start Chroma: `docker compose -f docker-compose.chroma.yml up -d`
3. Set `.env` as in **2b** (`MODEL_PROVIDER=azure_openai`, Foundry vars,
   `CHROMA_HOST=127.0.0.1`). Set `vector_store.backend: chroma` in `config.yaml`.
4. `pip install -e ".[store,governance]"`, then `check_model_host` / `ask` / `chat`
   as above.

Do **not** run large Ollama models on a small CPU VM. Prefer Azure for generation.
Step-by-step notes: [`azurereadme.md`](azurereadme.md).

### Smoke tests (no model required)

```bash
python scripts/smoke_retrieval.py
pytest -q
```

### Embeddings

Both ingest and retrieval embed with `fastembed` (`BAAI/bge-small-en-v1.5`) —
core dependency, ~130MB, CPU-only, no API key. Same library both sides so
vectors stay comparable. Falls back to a hash embedding (`HASH-FALLBACK`) only
if `fastembed` fails to import.

### Docker — Heinzy app image (S2)

```bash
docker build -t heinzy .
docker run --rm heinzy          # runs the retrieval smoke test
```

Build installs the `store` extra and pre-warms the `fastembed` cache. No model
weights are baked in — set `MODEL_PROVIDER` / Azure or Ollama env vars when you
run generation in the container.

### Shared Chroma host (S4) — optional team DB

**One shared Chroma Docker service**; clients set `CHROMA_HOST`. Default in
`config.yaml` is `chroma`; use `memory` for offline work.

| Role | What they do |
|------|----------------|
| **Host operator** | Runs [`docker-compose.chroma.yml`](docker-compose.chroma.yml), opens firewall TCP **8000** (Private) if teammates connect over LAN |
| **Everyone else** | `pip install -e ".[store]"`, set `CHROMA_HOST=<host-ip>`, `vector_store.backend: chroma` |

```bash
docker compose -f docker-compose.chroma.yml up -d
curl http://127.0.0.1:8000/api/v2/heartbeat
python scripts/smoke_store.py
```

Do not expose `:8000` to the public internet.

### Ask a real question end to end

With `.env` configured (Ollama **or** Azure) and a PDF in `data/corpus/`:

```bash
python scripts/ask_handbook.py --query "how many electives can I take?"
python scripts/chat.py   # interactive
```

First run ingests; later runs against the same store skip re-ingest when
`store.has_doc()` is true (Chroma). Tip: include **MISM** in the question when
retrieval feels weak — embeddings match handbook wording.

---

## Grounded answering & abstention

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

Layer 1 is evaluated by a policy engine rather than by an inline condition, so
the decision is auditable alongside every other governed action. Two engines
implement the same rule, chosen by `generation.policy.engine`.

`builtin` is the deterministic count check. No extra dependency, always
available, and what a plain `pip install -e .` gives you.

`agt` registers the same condition with the Microsoft Agent Governance Toolkit
as a `PolicyRule` whose validator returns False to deny, and the allow or deny
lands in the kernel's audit log as `request_denied` with reason
`policy_violation`. Install it with `pip install -e ".[governance]"`.

Asking for `agt` without the extra installed **denies every request** rather
than falling back to `builtin`. A safety gate that silently downgrades when its
dependency is missing still reports as governed while enforcing nothing, which
is worse than the plain `if` it replaced.

Layer 2 lives in [`heinzy/generation/abstain.py`](heinzy/generation/abstain.py),
deliberately apart from Layer 1, because the two fail differently. Layer 1 is
structural and replayable. Layer 2 is a model judgement that changes with the
model and has to be measured per model rather than assumed.

Layer 2 is not optional. Nearest neighbour search always returns *k* chunks and
has no way to report "no match", so every question retrieves something that
scores respectably. Measured on the real handbook with
`scripts/calibrate_floor.py`.

```
lowest in-corpus top score  : 0.7698   (ic-06, capstone project)
highest out-of-corpus score : 0.7449   (ooc-07, an invented transfer policy)
Separable. Suggested retrieval.score_floor: 0.757  (margin 0.0249)
```

Separable, but by 0.0249 across 16 questions. `score_floor` is set to **0.75**,
inside that window, which makes Layer 1 a live gate rather than a formality. On
the real handbook it now refuses all ten out-of-corpus questions before the
model is called at all, and still answers all six controls.

Be clear about what that buys and what it costs. The gain is that refusal no
longer depends on the model complying with an instruction, and an out-of-corpus
question never reaches generation. The cost is that the margin is thin. A
legitimate question scoring 0.74 will be refused, and 16 questions is a small
sample to place a threshold on. Widen the question set before trusting it
further, and re-run the calibration after any change to chunking, `k`, or the
embedding model.

One consequence worth knowing. With the floor live, Layer 1 now catches every
out-of-corpus question in the set, so Layer 2 never fires during the eval. It
is still covered by unit tests, and it still matters for the case the floor
cannot see, which is a question that retrieves genuinely high-scoring chunks
that happen not to answer it.

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
`config_hash=77179cb968f0`, k=5, temperature 0.

| Model | Out-of-corpus refused | Controls answered | Unsupported citations | Contentless answers |
|-------|----------------------|-------------------|-----------------------|---------------------|
| `gemma3:12b`, the configured model | **10/10** | 6/6 | 0 | 0 |
| `gemma2:9b` | **10/10** | 6/6 | 0 | 0 |
| `gemma4:e2b` | **10/10** | 6/6 | 0 | 0 |
| `llama3.2:latest` | **9/10** | 6/6 | 0 | 0 |

`gemma3:12b` is what `config.yaml` declares, so that row describes shipped
behaviour. It ran with no `MODEL_TAG` override, and every one of its six answers
carried a citation that resolved against the retrieved set.

The single `llama3.2` miss is `ooc-04`, where it deflects to Career Services
rather than inventing a policy. Not a refusal, but not the fabrication the
criterion targets either. `gemma4:e2b` refuses it outright.

Two reported but non-failing signals guard against passing on a technicality.

- **contentless answers**, where a bare citation with no prose is not an answer
  but is not a refusal either, so it would otherwise score as a success.
- **answers citing nothing**, where the answer may be perfectly grounded but
  with no citation to check, the grounding check passed on an empty set. A clean
  run is not the same as a verified one.

### Checking the generation host

```bash
python scripts/check_model_host.py
```

- **Ollama:** checks `MODEL_ENDPOINT` and that `model.tag` / `MODEL_TAG` is listed.
- **Azure:** checks Foundry/OpenAI credentials and deployment (see `azurereadme.md`).

Exits non-zero when the host is unreachable or misconfigured.

### Running it in the container

The same eval runs inside the Docker image, which is what makes the result a
property of the shipped artifact rather than of one laptop.

```bash
docker build -t heinzy .
docker run --rm \
  -e MODEL_PROVIDER=ollama \
  -e MODEL_ENDPOINT=http://host.docker.internal:11434 \
  -v "$(pwd)/data/corpus:/app/data/corpus:ro" \
  heinzy python scripts/eval_abstention.py --backend memory
```

For Azure generation in Docker, pass `MODEL_PROVIDER=azure_openai` and the
`AZURE_OPENAI_*` variables instead of `MODEL_ENDPOINT`. The corpus is mounted
rather than baked in (see `.dockerignore`).

Verified on the real handbook with `gemma3:12b`, ten of ten refused and six of
six answered, matching the host run exactly at `config_hash=a576a8c2c267`.

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

Neither is a generation bug, but both degrade grounded answering and belong to other
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
    generator.py          #   chunks -> grounded answer via Ollama or Azure
    grounding.py          #   citation extraction + retrieved-set check
    abstain.py            #   Layer 2, the insufficient-context sentinel
    policy.py             #   Layer 1 as a policy rule, builtin or AGT engine
  eval/
    abstention.py         #   out-of-corpus refusal eval + fixture store
  pipeline.py             # shared ingest-and-populate-store orchestration
eval/
  abstention_questions.yaml  # 8 out-of-corpus + 5 in-corpus control questions
  fixture_corpus.yaml        # labelled stand-in corpus (NOT the real handbook)
scripts/
  ask_handbook.py         # ingest + retrieve + generate, one question via --query
  chat.py                 # same, interactive
  eval_abstention.py      # proves the refusal claim; non-zero exit on failure
  check_model_host.py     # Ollama or Azure generation host health check
  calibrate_floor.py      # is a score floor viable on this corpus?
  smoke_retrieval.py      # placeholder-chunk demo, no real data needed
  smoke_store.py          # store factory smoke (memory or chroma)
tests/test_retrieval.py      # locks the retrieval contract
tests/test_store.py          # locks S4 get_store / adapter contract
tests/test_generation_grounding.py  # locks the refusal + citation contract
tests/test_policy_abstain.py        # locks the Layer 1 policy + fail-closed contract
azurereadme.md               # Azure AI Foundry / OpenAI runbook
docker-compose.ollama.yml    # optional local/LAN Ollama (GPU recommended)
docker-compose.chroma.yml    # shared Chroma host for the team (S4)
docker-compose.heinzy.yml    # app + mount feature/governance-policies src/policies (PEP)
heinzy/governance/           # loader for mounted OllamaGovernanceInterceptor
heinzy/tools/                # tool runners behind the PEP
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
| Docker (S2) | ✅ done | pinned base + deps |
| Generation host (S3) | ✅ Azure or local Ollama | \MODEL_PROVIDER=azure_openai\ or \ollama\; see [\zurereadme.md\](azurereadme.md) |
| Generation/answering (A3) | ✅ done | grounded answers via configured provider |
| Grounded answering + abstention (A3) | ✅ done | two-layer refusal, citation check, eval harness |
| Citations (A4) | 🟡 checked, not rendered | cited sections verified against the retrieved set; provenance flows from retrieval |
| Event log (A5) | ⬜ not started | append-only audit log (out of scope on this branch) |
| Eval harness (A6) | ⬜ not started | owner: Lisa |
| Governance layer | 🟡 mount-ready | PEP (`OllamaGovernanceInterceptor`) on `feature/governance-policies` — mount via [`docker-compose.heinzy.yml`](docker-compose.heinzy.yml); tool runners gated on this branch |

### Governance PEP mount (Docker)

The governance branch is **not** merged or edited from here. Mount a worktree at runtime:

```bash
git worktree add ../AIGovernance-governance feature/governance-policies

docker compose -f docker-compose.heinzy.yml build
GOVERNANCE_HOST_PATH=../AIGovernance-governance docker compose -f docker-compose.heinzy.yml run --rm heinzy
```

Inside the container: `GOVERNANCE_SRC=/governance/src` and `GOVERNANCE_POLICY_PATH=/governance/policies/governance_policy.yaml`. When those are set, `Generator` advertises tools and every tool call goes through `OllamaGovernanceInterceptor.evaluate_tool_call` before stub runners in `heinzy/tools/`.

Local (no Docker): set `GOVERNANCE_SRC` / `GOVERNANCE_POLICY_PATH` in `.env` to the same worktree paths (see `.env.template`).

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

Copy `.env.template` to `.env` and fill values. Never commit `.env`.

| Goal | Set |
|------|-----|
| Local Ollama | `MODEL_PROVIDER=ollama`, `MODEL_ENDPOINT=http://localhost:11434` |
| Azure generation | `MODEL_PROVIDER=azure_openai` + `AZURE_OPENAI_ENDPOINT` / `_API_KEY` / `_DEPLOYMENT` |
| Chroma on this machine | `CHROMA_HOST=127.0.0.1` and `vector_store.backend: chroma` |
| Offline store | `vector_store.backend: memory` (no Docker) |

Retrieval smoke tests do not need a live model. Generation does — run
`python scripts/check_model_host.py` before `ask` / `chat`.

Details for Azure: [`azurereadme.md`](azurereadme.md).
