# Eval set uses fabricated placeholder content, not the real MISM handbook

**Labels:** bug

## What
`eval/questions.jsonl` — the question set the A6 harness scores against — is
seeded with **invented placeholder facts** (e.g. "144 units to graduate", "up to
4 electives"), not content verified against the actual MISM handbook. The
`expected_section_contains` markers ("Electives", "Graduation", etc.) are
likewise guesses at section names, not the real section paths the ingestion
pipeline produces from the handbook PDF.

As a result, `python eval/run_eval.py` currently measures the harness against
fiction. Retrieval hit@k / MRR are computed against section names that may not
exist in the real corpus, and answer-correctness is graded against ground-truth
answers nobody confirmed are true.

## Why
- A6's "done when" is explicitly a set with **advisor-written ground truth**,
  including out-of-corpus items. Placeholder content does not satisfy that — a
  passing score today says nothing about real retrieval quality.
- The PRD flags Context Entity Recall as the known weak metric to tune via
  chunk size / overlap / k. You cannot tune against placeholder sections; the
  tuning signal is only meaningful on the real handbook.
- Green eval numbers on fabricated data are worse than no numbers, because they
  look like validation.

## Requirements
- Replace `eval/questions.jsonl` with 20–30 questions whose answers are verified
  against the real MISM handbook, written or reviewed by an advisor (Goutam /
  Academic Services per the PRD dependency), including deliberate out-of-corpus
  items.
- Set `expected_section_contains` to real section paths emitted by the ingestion
  pipeline for the actual handbook (confirm against `structure.build_sections`
  output), not guessed names.
- Re-run the harness against the real corpus in Chroma and record a baseline
  result file (`eval/results/eval_<config_hash>_<ts>.json`).
- Owner per PRD: Lisa (harness) + advisor input for ground truth.

## IMPORTANT
- Closed only after end-to-end testing in the cloud with Docker and it ran
  correctly. If infra isn't set in the cloud, ping @asriram15 until solved and
  retry.
- If a bug is found, create a new ticket in this format and ping it in chat.
- On close, open a PR explaining what/why and any issues.

## Expected outcome
`eval/run_eval.py` scores the pipeline against real, advisor-verified handbook
Q&A, producing a baseline that actually reflects retrieval and answer quality.

## QA
Open `eval/questions.jsonl`; confirm every in-corpus answer resolves to a real
passage in the handbook and every `expected_section_contains` matches a section
the pipeline actually emits. Run the harness against the real Chroma corpus and
confirm the result file is written with a non-trivial hit@k.
