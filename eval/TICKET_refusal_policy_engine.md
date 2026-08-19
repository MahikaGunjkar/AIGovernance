# Refusal/abstention is prompt-only + a display heuristic, not a policy engine

**Labels:** bug

## What
Heinzy's "refusal" behavior — declining to answer when the handbook doesn't
cover a question — is currently produced two ways, neither of which is an actual
governance/policy layer:

1. **Generation side:** the system prompt in `heinzy/generation/generator.py`
   instructs Gemma to say it can't answer if the excerpts don't contain the
   answer. This is prompt engineering — it depends on the model complying, and a
   probabilistic model will sometimes not comply (hallucinate an answer anyway,
   or decline when it shouldn't).
2. **Display side:** `_looks_declined()` in `heinzy/webui/app.py` (and the twin
   `_looks_abstention()` in `eval/run_eval.py`) does a keyword scan of the
   answer text to decide whether to render it as "declined". This is a
   post-hoc string heuristic on the model's output, not an enforced decision.

There is no deterministic control that *intercepts* a low-relevance retrieval
and forces an abstain regardless of what the model says. The PRD's whole premise
(runtime governance, deterministic policy enforcement for every action) is not
yet met for the one policy we most rely on.

## Why
This matters because:
- The release criteria require the system to abstain rather than fabricate on
  out-of-corpus questions, "verified with deliberate out-of-scope questions."
  Right now that guarantee rests on prompt compliance + a keyword match, both of
  which can silently fail.
- An advisor could be shown a fabricated answer that reads confident and does
  *not* trip any decline cue, so the UI renders it as a normal (trusted) answer.
- The keyword heuristic is brittle: a legitimate answer containing the word
  "outside" or "not available" gets mis-flagged as declined, and a fabrication
  phrased confidently gets flagged as answered.

## Requirements
- A deterministic abstention gate that runs BEFORE/AROUND generation, based on
  retrieval signal (e.g. top-k score below `retrieval.score_floor`, or no hits),
  independent of the model's wording.
- When the gate fires, the system returns a structured "declined" state (a flag
  on the response object), not a sentence the UI has to keyword-scan.
- The UI and eval harness consume that structured flag instead of
  `_looks_declined` / `_looks_abstention` keyword scans.
- Config-driven thresholds (no constants in source, per S5).
- Keep the generation-side prompt instruction as defense in depth, but stop
  relying on it as the mechanism.

## IMPORTANT
- Closed only after end-to-end testing in the cloud with Docker and it ran
  correctly. If infra isn't set in the cloud, ping @asriram15 until solved and
  retry.
- If a bug is found, create a new ticket in this format and ping it in chat.
- On close, open a PR explaining what/why and any issues.

## Expected outcome
Out-of-corpus questions are declined deterministically based on retrieval
signal, the decline state is a structured field rather than an inferred one, and
the behavior no longer depends on model compliance or keyword matching.

## QA
Ask several out-of-corpus questions phrased to *avoid* decline keywords;
confirm the system still declines them. Ask an in-corpus question phrased to
*contain* a decline keyword; confirm it is NOT mis-flagged as declined.
