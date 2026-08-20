# Heinzy — Governed Advising Assistant

A governed internal assistant for Heinz College advisors. It answers program and
policy questions from official MISM handbook documents, returns verifiable
citations, and can run inside a governance layer that constrains what it can
access and do.


## How to run

| Mode | App runs on | LLM runs on |
|------|-------------|-------------|
| Local + Ollama | Laptop | Local Ollama |
| Local + Azure | Laptop | Azure AI Foundry / OpenAI |
| VM + Azure | Linux VM | Azure AI Foundry / OpenAI |

Retrieval always runs with the app. Only **generation** calls the model.  
Azure detail + troubleshooting: [`docs/azure.md`](docs/azure.md).

**Note:** Layer 1 abstention and citations work on Azure. Governed **tool**
calls (web search / write tools behind the interceptor) are Ollama-only until
Azure tool calling is wired — see [`docs/azure.md`](docs/azure.md#governance-tools-on-azure).

### 1. Clone, governance worktree, install

```bash
git clone <this-repo> && cd AIGovernance

# Governance PEP (separate branch — mount at runtime; do not merge it here)
git worktree add ../AIGovernance-governance feature/governance-policies

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -e ".[store,dev,governance,webui]"
cp .env.template .env
mkdir -p data/corpus    # put the handbook PDF here
```

In `.env`, point at the worktree (paths relative to the repo root):

```
GOVERNANCE_SRC=../AIGovernance-governance/src
GOVERNANCE_POLICY_PATH=../AIGovernance-governance/policies/governance_policy.yaml
GOVERNANCE_HOST_PATH=../AIGovernance-governance
```

Skip the worktree only if you do not need the tool-governance PEP (chat still
works; tool calls stay off).

### 2. Choose generation + store

**Ollama (local):** install [Ollama](https://ollama.com/), `ollama pull gemma3:12b`, then in `.env`:

```
MODEL_PROVIDER=ollama
MODEL_ENDPOINT=http://localhost:11434
```

**Azure (laptop or VM):** in `.env` (full steps: [`docs/azure.md`](docs/azure.md)):

```
MODEL_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.services.ai.azure.com
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
CHROMA_HOST=127.0.0.1
```

Governed tool calls stay off on Azure; handbook chat + abstention still work.

**Store:** `config.yaml` → `vector_store.backend: memory` (no Docker), or `chroma` plus:

```bash
docker compose -f docker-compose.chroma.yml up -d
```

On a VM, prefer **chroma + Azure** (not large Ollama on CPU).

### 3. Run

```bash
python scripts/check_model_host.py
python scripts/ask_handbook.py --query "What does the MISM curriculum cover?"
python scripts/chat.py
```

**Web UI** (browser chat at http://localhost:5000):

```bash
pip install -e ".[webui]"
python -m heinzy.webui.app
```

Uses the same `.env` / store / model as the CLI (Ollama or Azure). First load
may ingest the PDF. Optional Docker UI (passes Azure env from `.env` too):
`docker compose -f docker-compose.webui.yml up --build`.

```bash
# optional checks
python scripts/smoke_retrieval.py
pytest -q
```

Docker app image / optional LAN Ollama / team Chroma: see
[`docker-compose.heinzy.yml`](docker-compose.heinzy.yml) (governance mount),
[`docker-compose.chroma.yml`](docker-compose.chroma.yml),
[`docker-compose.ollama.yml`](docker-compose.ollama.yml).

Tip: include **MISM** in questions when retrieval feels weak.

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

### Checking the generation host

```bash
python scripts/check_model_host.py
```

- **Ollama:** checks `MODEL_ENDPOINT` and that `model.tag` / `MODEL_TAG` is listed.
- **Azure:** checks Foundry/OpenAI credentials and deployment (see `docs/azure.md`).

Exits non-zero when the host is unreachable or misconfigured.

### Running it in the container


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


### Governance PEP (Docker)

Same worktree as in [How to run](#1-clone-governance-worktree-install). With Docker:

```bash
docker compose -f docker-compose.heinzy.yml build
GOVERNANCE_HOST_PATH=../AIGovernance-governance docker compose -f docker-compose.heinzy.yml run --rm heinzy
```

### Swapping vector-store backends

Retrieval talks only to the `VectorStore` protocol. For Chroma: run
`docker-compose.chroma.yml`, set `CHROMA_HOST` in `.env`, and
`vector_store.backend: chroma` in `config.yaml`.


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

Details for Azure: [`docs/azure.md`](docs/azure.md).
