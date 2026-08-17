# PR: Web UI for testing the RAG chat (closes #17, #18)

## What
Adds a lightweight Flask web UI that wraps the existing RAG pipeline so anyone
can open a browser, ask questions, and see a running conversation with answers
and citations — no CLI or env-var knowledge needed to *use* it.

- `heinzy/webui/app.py` — Flask app. Builds `ingest_and_populate_store` +
  `Retriever` + `Generator` **once** at startup (same as `chat.py`), then serves
  `/api/ask` which does one retrieve + generate per question. **No new pipeline
  logic** — it's a frontend over the exact call pattern in `ask_handbook.py`.
- `heinzy/webui/templates/index.html` — single-page chat UI.
- `docker-compose.webui.yml` — runs the UI in the existing image, wired to the
  shared Chroma + Gemma hosts via the same `CHROMA_HOST` / `MODEL_ENDPOINT` /
  `MODEL_TAG` env vars the CLI uses.
- `tests/test_webui.py` — 6 tests, run without live infra via a fake engine.
- `webui` extra added to `pyproject.toml` (Flask only); Dockerfile installs it.

## Why
Closes #17: the only end-to-end test path was `scripts/chat.py`, which needs
terminal access, env vars, and reading raw text — a real barrier for Goutam.
Closes #18: chat history now persists in the browser across refreshes.

## Ticket #17 requirements — how each is met
- **Minimal local web UI wrapping existing pipeline, no new pipeline logic** —
  `app.py` imports and calls the existing modules only; all RAG logic is
  unchanged.
- **Visible chat history in the session** — each Q/A pair renders as a turn.
- **Each turn independently retrieved/generated, no multi-turn memory** — the
  model never sees prior turns; the UI is a visual log, matching `chat.py`.
- **Show cited sources (section_path, source_pages, score)** — rendered under
  each answer, the same fields `chat.py` prints.
- **Declined answers look visibly different** — declined turns render in an
  amber "declined" style with a distinct label.
- **Same env-var config as CLI** — `CHROMA_HOST` / `MODEL_ENDPOINT` /
  `MODEL_TAG`, nothing new.
- **History in-browser, not a database** — see #18.

## Ticket #18 requirements — how each is met
- **Store each Q/A pair (with citations + decline state) in localStorage** —
  every turn is persisted on send.
- **Restore on page load** — history rehydrates from localStorage on load.
- **Per-browser, not shared / not a backend** — pure client-side localStorage;
  a Clear button empties it.

## Arising issues / bugs found
Per the ticket process, filing two bugs found while building this (drafts in
`eval/TICKET_*.md`, pinging in chat for assignment):

1. **Refusal logic is prompt-only + a display heuristic, not a policy engine.**
   The UI's "declined" styling relies on `_looks_declined()`, a keyword scan of
   the model's output — the same brittle approach as the eval harness. There is
   no deterministic gate that forces abstention on low-relevance retrieval. This
   is a governance gap against the PRD's core premise. → `TICKET_refusal_policy_engine.md`
2. **Eval set is fabricated placeholder content, not the real handbook.** The A6
   question set uses invented facts and guessed section names, so eval scores
   don't reflect real retrieval quality. → `TICKET_eval_real_handbook.md`

Neither blocks this UI PR, but both should be picked up before any of these
numbers or decline states are treated as trustworthy.

## Testing
- `pytest tests/test_webui.py` — 6/6 pass locally (fake engine, no infra).
- **Cloud E2E with Docker: NOT yet run from this branch.** Per the ticket, #17
  and #18 close only after the whole app is tested in the cloud with Docker.
  That requires the shared Chroma + Gemma hosts reachable and the handbook
  indexed. **Pinging @asriram15** to confirm cloud infra is up, then I'll run
  `docker compose -f docker-compose.webui.yml up --build` against it and confirm
  the QA script (ask 3–4 questions incl. one declined, verify citations +
  decline state, refresh to confirm persistence) before closing.

## How to run
```bash
pip install -e ".[webui]"
python -m heinzy.webui.app          # http://localhost:5000
# or in Docker, wired to shared hosts:
docker compose -f docker-compose.webui.yml up --build
```
