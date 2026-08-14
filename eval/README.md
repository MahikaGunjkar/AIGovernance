# Evaluation harness (A6)

Scores the Heinzy RAG pipeline against a curated question set and writes a
result file tagged with the config hash and model build, so any run is
reproducible.

## Run

```bash
# Retrieval only — fast, no Gemma/Ollama needed:
python eval/run_eval.py --no-generation

# Full: retrieval + generation + answer grading (needs Gemma + store reachable):
python eval/run_eval.py

# Grade answers with the LLM instead of embedding similarity:
python eval/run_eval.py --judge ollama

# Tuning:
python eval/run_eval.py --k 3 --sim-threshold 0.6 --questions eval/questions.jsonl
```

Prereqs are the same as `scripts/ask_handbook.py`: a populated corpus in
`data/corpus/` and (for generation) a reachable `MODEL_ENDPOINT`. With
`--no-generation` you only need retrieval to work.

## Metrics

| Metric | Meaning |
|--------|---------|
| `hit_at_k` | fraction of in-corpus questions where a chunk from the expected section made the top-k |
| `mrr` | mean reciprocal rank of the first correct-section hit |
| `answer_correctness` | fraction of in-corpus answers scored correct (see grader) |
| `abstention_rate` | fraction of out-of-corpus questions the system correctly declined |

## Grading answers without an API key

Default grader is **embedding similarity**: cosine between the model answer and
the ground-truth answer, using the same local `Embedder` the retriever uses.
Zero new dependencies, instant, deterministic. An answer counts as correct when
similarity ≥ `--sim-threshold` (default 0.55 — tune against your set).

`--judge ollama` instead asks the shared Gemma host for a 0–1 score. More
faithful to meaning, but slower and non-deterministic — use it for a final
check, not every iteration.

Out-of-corpus questions are graded on **abstention**: the system passes if it
declines rather than fabricating. This is what verifies the grounding guarantee.

## Question set format

`eval/questions.jsonl`, one JSON object per line:

```json
{"id": "q01", "question": "...", "expected_answer": "...", "expected_section_contains": "Electives", "in_corpus": true}
```

- `expected_section_contains` — substring matched against each hit's
  `section_path` for the retrieval hit@k / MRR check. Use `null` for
  out-of-corpus items.
- `in_corpus: false` — the answer isn't in the handbook; the system should
  abstain.

The starter set is small and uses placeholder MISM facts. **Replace it with
advisor-written Q&A** (20–30 items including out-of-corpus) for a real
evaluation — that's the A6 "done when".

## Output

Results land in `eval/results/eval_<config_hash>_<timestamp>.json` with a
`summary` block and per-question detail. The filename and payload both carry the
`config_hash`, so a result is always traceable to the exact config that produced
it.
