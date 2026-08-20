# Evaluation harness (A6)

Two harnesses live here. Prefer the abstention suite for grounding guarantees;
use `run_eval.py` for retrieval hit@k / MRR against a curated JSONL set.

## Primary: abstention / grounding (`scripts/eval_abstention.py`)

Uses `eval/abstention_questions.yaml` (handbook-verified out-of-corpus +
in-corpus controls). See [README → Running the eval](../README.md#running-the-eval).

```bash
python scripts/eval_abstention.py --fixture
python scripts/eval_abstention.py
python scripts/calibrate_floor.py
```

Works with `MODEL_PROVIDER=ollama` or `azure_openai` (same `.env` as chat).

## Secondary: retrieval + answer grading (`eval/run_eval.py`)

Scores hit@k / MRR / answer correctness against `eval/questions.jsonl` and
writes `eval/results/eval_<config_hash>_<timestamp>.json`.

```bash
# Retrieval only — fast, no generation host needed:
python eval/run_eval.py --no-generation

# Full: retrieval + generation + answer grading:
python eval/run_eval.py

# Grade answers with the LLM instead of embedding similarity:
python eval/run_eval.py --judge ollama

# Tuning:
python eval/run_eval.py --k 3 --sim-threshold 0.6 --questions eval/questions.jsonl
```

Prereqs match `scripts/ask_handbook.py`: corpus in `data/corpus/` and (for
generation) a reachable model via `.env`. With `--no-generation` you only need
retrieval.

### Metrics

| Metric | Meaning |
|--------|---------|
| `hit_at_k` | fraction of in-corpus questions where a chunk from the expected section made the top-k |
| `mrr` | mean reciprocal rank of the first correct-section hit |
| `answer_correctness` | fraction of in-corpus answers scored correct (see grader) |
| `abstention_rate` | fraction of out-of-corpus questions the system correctly declined |

### Grading answers without an API key

Default grader is **embedding similarity**: cosine between the model answer and
the ground-truth answer, using the same local `Embedder` the retriever uses.
An answer counts as correct when similarity ≥ `--sim-threshold` (default 0.55).

`--judge ollama` asks the configured generation host for a 0–1 score (works
with Ollama; Azure users should stick to the embedding grader or abstention
harness).

### Question set format

`eval/questions.jsonl`, one JSON object per line:

```json
{"id": "q01", "question": "...", "expected_answer": "...", "expected_section_contains": "Electives", "in_corpus": true}
```

The starter JSONL still uses placeholder MISM facts. For release-quality A6
numbers, replace it with advisor-written Q&A (20–30 items including
out-of-corpus). The abstention YAML is already handbook-verified.
