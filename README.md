# CineMatch

An AI agent that recommends **movies** personalized to your taste and verified as
available to stream/rent/buy on a specific platform in a specific country.

It combines a **RAG pipeline** over an IMDB movie dataset (Pinecone) for
taste-matching with the **TMDB API** for live availability verification — so it
never claims a movie is available without a real lookup. Movies only, no TV.

---

## Architecture

A deterministic orchestrator drives a 3-stage pipeline with a bounded
ReAct↔Reflection retry loop:

```
User Input
  → TasteExtractor      (Stage 0 — parse request into structured preferences)
  → ReActAgent          (Stage 1 — agentic retrieval: rag_search /
                          tmdb_fallback_search / ask_user_clarification)
  → ReflectionAgent     (Stage 2 — verify_recommendation / ask_user_clarification)
       ├─ approve  → composes final response → return
       ├─ reject   → feed critique back to ReActAgent (loop, max MAX_PASSES)
       └─ clarify  → return question to the user
```

Module names are identical across the code, the `steps` trace, and the
architecture diagram: `TasteExtractor`, `ReActAgent`, `ReActAgent/rag_search`,
`ReActAgent/tmdb_fallback_search`, `ReActAgent/ask_user_clarification`,
`ReflectionAgent`, `ReflectionAgent/verify_recommendation`,
`ReflectionAgent/ask_user_clarification`.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/team_info`          | Student/team details |
| GET  | `/api/agent_info`         | Description, purpose, prompt template, examples |
| GET  | `/api/model_architecture` | Architecture diagram (PNG) |
| POST | `/api/execute`            | Main entry point — returns response + full step trace |

`POST /api/execute` body:

```json
{ "prompt": "Something like Sicario on Netflix in Israel", "conversation_history": [], "prior_state": null }
```

`prior_state` (optional) and the response's `state` round-trip cross-turn memory
— movies already ruled out or already recommended — since the server itself is
stateless between calls. Send `null`/omit on the first turn, then echo back
whatever `state` the previous response returned on every following turn. See
**Agentic loop overhaul** below for why this exists.

**Interactive vs. non-interactive mode.** Whether the agent may ask a clarifying
question is inferred from `conversation_history`:

- **Present** (a list, even empty `[]`) → **interactive**. A GUI sends this every
  turn, so `ask_user_clarification` is offered to the agents and the response may be
  a follow-up question instead of recommendations.
- **Omitted** (`null`) → **non-interactive**. Automated evaluators / raw API callers
  cannot answer questions, so the clarification tool is withheld and the pipeline
  **always** returns movie recommendations, making sensible assumptions for any
  missing detail (e.g. country/platform).

`session_id` is optional (used only for best-effort Supabase logging).

---

## Run locally

```bash
python -m venv .venv ; .\.venv\Scripts\Activate.ps1   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env        # then fill in real keys (see below)
uvicorn api.index:app --reload
```

Open <http://localhost:8000/> for the chat UI (see **Frontend (UI)** below), or
<http://localhost:8000/docs> for interactive Swagger, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Slow-burn crime like Sicario, on Netflix in Israel"}'
```

> Requires Python 3.12 (matches the Vercel runtime).

---

## Frontend (UI)

A minimal, dependency-free chat UI lives at `public/index.html` — a single self-contained
file (inline CSS + vanilla JS, no build step). It's served two ways:

- **Locally**: `GET /` in `api/index.py` returns `public/index.html` directly, so
  `uvicorn api.index:app --reload` + opening <http://localhost:8000/> is enough.
- **On Vercel**: files under `public/` are served automatically as static assets at `/`,
  alongside the `/api/*` serverless function — no extra config needed.

What it does:

- Sends every turn to `POST /api/execute` with `conversation_history` (starting as `[]`),
  which keeps the pipeline in **interactive mode** so it can ask clarifying questions.
- Renders the agent's reply as markdown, and — for each step in the response's `steps[]` —
  a collapsible entry showing that step's `module`, `system_prompt` / `user_prompt`, and
  `response` (click **Show execution trace** under any reply to expand it).
- Detects a clarifying question (the last trace step's `module` ends with
  `ask_user_clarification`) and just leaves the input open for your reply; otherwise it
  shows **Satisfied** (ends the chat client-side, no API call) and **Try again** (lets you
  type what was wrong and reruns the pipeline with that as feedback) — matching the
  stop-condition design in **GUI plan** below.

### Testing the UI locally

1. `uvicorn api.index:app --reload` (see **Run locally** above; `.env` needs at least
   `LLMOD_API_KEY`, `LLMOD_BASE_URL`, `LLMOD_MODEL`, `LLMOD_EMBEDDING_MODEL`, `TMDB_API_KEY`).
2. Open <http://localhost:8000/> in a browser.
3. Send a prompt (e.g. *"Slow-burn crime like Sicario, on Netflix in Israel"*) and confirm:
   - A reply renders, and **Show execution trace** expands to show each step.
   - **Satisfied** ends the conversation and disables the input, with **no** network
     request (check the browser's Network tab).
   - **Try again** re-calls `/api/execute` with your typed complaint appended.
4. Send a vague prompt (e.g. *"recommend me something"*) to trigger a clarifying question;
   confirm it renders as a normal reply, the input stays open, and your answer is appended
   to `conversation_history` on the next request (inspect the request body in the Network
   tab — the history array should grow each turn).

---

## Environment variables

Set these (locally in `.env`, and in the Vercel project settings for production):

```
LLMOD_API_KEY, LLMOD_BASE_URL, LLMOD_MODEL, LLMOD_EMBEDDING_MODEL
TMDB_API_KEY
```

Optional (the app runs fully without these):

```
PINECONE_API_KEY, PINECONE_INDEX_NAME   # only needed if RAG_BACKEND=pinecone
SUPABASE_URL, SUPABASE_KEY              # run-logging is best-effort
RAG_BACKEND                             # "local" (default) or "pinecone"
RAG_DOCUMENTS_PATH                      # default: data/processed/rag_documents.jsonl
```

Optional tunables: `MAX_PASSES`, `MAX_TOOL_CALLS`, `RAG_TOP_K`, `MIN_CANDIDATES`,
`MAX_CANDIDATES`, `RAG_BUDGET_BY_PASS` (default `-1,1,0` — unlimited
`rag_search` calls on pass 1, at most 1 on pass 2, none from pass 3 on). See
`.env.example`.

---

## RAG index

`rag_search` reads from either a local JSONL file (`RAG_BACKEND=local`, the
default — no Pinecone account needed) or a Pinecone index (`RAG_BACKEND=pinecone`).
Both are built the same way, from `data/processed/canonical_movies.csv` only
(never `unmatched_movies.csv` — those rows have no verified `tmdb_id`).

Build the local index:

```bash
# Smoke test — no API key needed, no cost, exercises the full pipeline:
python scripts/build_rag_index.py --limit 20 --mock-embeddings

# Real embeddings for a small sample (uses LLMOD_API_KEY):
python scripts/build_rag_index.py --limit 20

# Full build (all 9,160 canonical movies):
python scripts/build_rag_index.py
```

This writes `data/processed/rag_documents.jsonl`, one JSON object per movie:
`tmdb_id, title, year, genres, overview, score, runtime, embedding`.

Verify every document has a real `tmdb_id`:

```bash
python -c "
import json
with open('data/processed/rag_documents.jsonl', encoding='utf-8') as f:
    docs = [json.loads(l) for l in f if l.strip()]
assert all(d.get('tmdb_id') for d in docs), 'found a document with no tmdb_id'
print(f'{len(docs)} documents, all with a valid tmdb_id.')
"
```

Run a local RAG smoke test (no Pinecone/API key required if built with
`--mock-embeddings`):

```bash
python -c "
import asyncio
from agent.tools import rag_search
result = asyncio.run(rag_search.execute({'query_text': 'dark action thriller like John Wick', 'top_k': 5}))
for r in result['results']:
    print(r['tmdb_id'], r['title'], r.get('local_score'))
"
```

To use Pinecone instead, set `RAG_BACKEND=pinecone` and `PINECONE_API_KEY` in
`.env`, then run `python scripts/build_rag_index.py --upsert-pinecone` (vector
ids are deterministic: `movie-{tmdb_id}`, so re-running is idempotent).

---

## Deploy (Vercel)

First time deploying to Vercel? Follow these steps in order.

1. **Protect your secrets before pushing.** A real `.env` with live API keys exists
   locally — make sure it's git-ignored and not committed:
   ```bash
   git check-ignore .env               # should print ".env" (ignored — good)
   git ls-files --error-unmatch .env   # should ERROR "did not match" (not tracked — good)
   ```
   If `.env` shows as tracked instead, add it to `.gitignore`, run
   `git rm --cached .env`, commit that, and rotate any keys that were ever pushed.
2. **Push this repo to GitHub** (create a repo on github.com if you don't have one yet,
   then `git push`).
3. **Import into Vercel.** Sign in at vercel.com with "Continue with GitHub", click
   **Add New… → Project**, select this repo, and click **Import**.
4. **Framework preset: leave it as "Other".** This project is a Python serverless
   function plus static files, not a Node framework — do **not** set a build command
   or output directory; `vercel.json` already handles routing.
5. **Add environment variables** (on the import screen, or later under
   **Project → Settings → Environment Variables**) for **Production** (and Preview):
   copy the required keys (`LLMOD_API_KEY`, `LLMOD_BASE_URL`, `LLMOD_MODEL`,
   `LLMOD_EMBEDDING_MODEL`, `TMDB_API_KEY`) and any optional ones you use, from your
   local `.env`. Never commit these values.
6. **Deploy.** Click **Deploy** and wait for the build to finish (green "Ready").
   `vercel.json` routes `/api/*` to the Python function (300 s max duration) and
   serves everything in `public/` — including `public/index.html` — as static files
   at `/`.
7. **Redeploys are automatic**: every push to the connected branch triggers a new
   deployment; no CLI required.
8. **Verify**: open `https://<your-app>.vercel.app/` for the chat UI, and check
   `GET /api/team_info`, `GET /api/agent_info`, `GET /api/model_architecture`, and a
   `POST /api/execute`.

---

## Next steps (before submission)

This repo is the **backend**. To make it fully production-ready:

1. **Architecture diagram** — drop your PNG at `public/architecture.png`. Until
   then `GET /api/model_architecture` returns a clear 404. Names in the diagram
   must match the module names listed above.
2. **Fill real credentials** in `.env` (and Vercel): LLMod.ai key, TMDB key,
   Supabase URL + key, Pinecone key + index name. The Pinecone index
   (`cinematch-movies`) is assumed to already exist and be populated.
3. **Create the Supabase tables** `pipeline_runs` and `conversations` (schemas in
   the build spec §9). Run logging is best-effort, so the app works even before
   these exist — but you need them for history/logging.
4. **Populate the TMDB lookups** in `agent/tmdb_mappings.py`
   (`PLATFORM_TO_PROVIDER_ID`, `COUNTRY_TO_REGION_CODE`) from TMDB's
   `/watch/providers/movie` and `/watch/providers/regions` endpoints.
5. **Replace placeholder team info** in `/api/team_info`
   (`api/index.py`) with real names, emails, and your batch/order number.
6. ~~Build the frontend~~ — done: `public/index.html` is a minimal chat page at `/`
   calling `POST /api/execute` and displaying `response` + the full `steps` trace, no
   auth guards. See **Frontend (UI)** and **GUI plan** below.

---

## GUI plan (frontend design — implemented in `public/index.html`)

The backend supports the frontend by design: sending
`conversation_history` on every turn puts the pipeline in **interactive** mode
(see *API endpoints* above), so the agent can ask a clarifying question when it
needs one. The planned chat UI:

1. **Every turn sends `conversation_history`** — starting as `[]` on the first
   message — so the backend always runs in interactive mode. Each user message and
   agent reply is appended to the history the client keeps.
2. **Clarifying questions** returned by the agent are rendered as a normal assistant
   turn. The user's reply is appended to `conversation_history` and re-sent to
   `/api/execute`, which reruns the pipeline with the added context.
3. **After recommendations are shown**, render two buttons under them:
   - **Satisfied** → display a fixed client-side template message (e.g.
     *"Happy to help — enjoy the movie! 🎬"*) and **end the conversation** (disable
     the input). This is purely frontend: **no `/api/execute` call and no LLM call.**
   - **Try again** → let the user type what was wrong, append that as a `user`
     message to `conversation_history`, and re-call `/api/execute` to rerun the
     pipeline with the complaint as feedback.

This keeps the "stop condition" entirely on the client — the satisfied path costs
nothing, and the try-again path reuses the existing multi-turn history mechanism.

---

## Project layout

```
api/index.py              FastAPI app — all /api/* routes
agent/orchestrator.py     Deterministic loop controller
agent/taste_extractor.py  Stage 0
agent/react_agent.py      Stage 1 (agentic retrieval loop)
agent/reflection_agent.py Stage 2 (critique + verify)
agent/tools/              rag_search, tmdb_fallback_search,
                          verify_recommendation, ask_user_clarification
agent/clients/            llm, tmdb, pinecone, supabase clients
agent/models.py           Pydantic data shapes
agent/prompts.py          All system prompts + user prompt template
agent/config.py           Env vars + tunable parameters
public/index.html         Minimal chat UI (frontend, served at /)
public/architecture.png   (you provide this)
```

---

## Agentic loop overhaul (2026-07-13)

The ReAct ↔ Reflection loop was slow and lost progress between passes. Changes made:

- **Availability check moved out of Reflection's LLM call.** `verify_recommendation`
  used to run as a tool the Reflection LLM called mid-conversation (LLM call →
  tool → second LLM call, per pass). The orchestrator now calls it directly as a
  plain async function right after each ReAct draft — deterministic, already
  parallel across candidates — and Reflection receives the pre-verified results.
  Reflection is now a single LLM call instead of 2-3 sequential round-trips.
  *Why:* availability is a deterministic lookup, not a judgment call; there's no
  reason to spend an LLM turn deciding to call a tool whose answer is binary.
- **Parallel tool execution within one ReAct turn.** When the model requests
  multiple tool calls in the same turn (e.g. two `rag_search` calls), they now
  run concurrently via `asyncio.gather` instead of one after another.
  *Why:* the trace showed this happening on nearly every pass; the calls are
  independent, so serializing them was pure wasted latency.
- **Approved/excluded state now carries across passes.** `ReactContext` gained
  `approved` and `excluded` lists; the Reflection verdict is now per-candidate
  (`approved_ids` / `rejected`, not just an overall approve/reject). Approved
  movies are saved toward the final answer and never re-searched; excluded ones
  (failed availability or bad taste match) are listed in ReAct's next prompt so
  it looks for genuinely different movies instead of re-surfacing the same
  rejected titles. The loop exits as soon as enough movies are approved,
  without waiting for a full extra pass.
  *Why:* previously each pass started from a blank slate — a rejected batch was
  simply discarded and ReAct had no memory of what didn't work, so it would
  often propose the same or very similar candidates again.
- **Both agents know when they're on the last pass.** `MAX_PASSES` went from 2
  to 3 (now that retries are cheap and targeted, an extra round is worth it).
  On the final pass, ReAct is told to return its strongest list immediately
  (no clarification), and Reflection is told "reject" is not an option — it
  must compose the best possible answer from whatever was approved, so the
  user always gets a real response instead of a dangling rejection.
  *Why:* previously a rejection on the last allowed pass fell through to a
  generic "best guess" composer with no availability confirmation, even when
  perfectly good approved movies existed from earlier passes.
- **`tmdb_fallback_search` upgraded to actually search "niche."** Added
  `vote_count_min`/`vote_count_max` (the main lever for avoiding blockbusters),
  `sort_by`, `keywords` (resolved via TMDB's `/search/keyword`), `runtime_min`/
  `runtime_max`, and `original_language`. Fixed the monetization filter to only
  count `flatrate|free|ads` (subscription-style availability) instead of also
  matching `rent|buy`, which was reporting pay-per-title movies as "available."
  Genre IDs are now mapped to names (matching `rag_search`'s output shape)
  instead of leaking raw TMDB integers.
  *Why:* the tool defaulted to `sort_by=popularity.desc` with no way to filter
  it out — a "niche rom-com" request was surfacing *Mamma Mia!* and *The 40-
  Year-Old Virgin*, the opposite of what was asked.
- **Cleaner execution trace.** ReAct no longer logs a step for a turn that is
  just `{"content": null, "tool_calls": [...]}` — the following
  `ReActAgent/<tool>` step already shows what happened. The prompt now asks the
  model to include a one-sentence rationale when it does call a tool, so the
  steps that remain are informative instead of empty placeholders.
  *Why:* the trace was cluttered with content-free entries that added noise
  without adding information.
- **Cross-turn memory now survives between conversation turns.** Previously
  `ReactContext.approved`/`excluded` only lived inside a single `run_pipeline`
  call — a "Try again" turn started from a blank slate and would re-draft,
  re-verify, and re-reject the exact same movies the *previous* turn already
  ruled out or already showed the user. Added `SessionState` (`excluded` +
  `recommended`), sent by the client as `prior_state` and returned as `state`
  on every response; the orchestrator seeds `context.excluded` from it before
  pass 1 and folds this turn's outcome back in on every return path. The GUI
  (`public/index.html`) now stores and echoes it automatically. As a
  belt-and-braces fallback for callers that don't round-trip `state`, the
  Taste Extractor prompt also scans `conversation_history` for movies the
  assistant already recommended and adds them to `exclude`.
  *Why:* the trace from a real "cute niche rom-com" run showed pass 3 of a
  single turn re-retrieving movies already excluded in pass 2 — the same
  problem, just worse, across conversation turns, since the whole pipeline
  state was thrown away between API calls.
- **Retrieval shifts from RAG to TMDB fallback as passes go on, deterministically.**
  Previously `tmdb_fallback_search` only unlocked when a Reflection verdict
  explicitly asked for it or every candidate failed availability — so a pass 2
  or 3 could still spend its whole budget on `rag_search`, whose index has no
  availability signal at all (in one real trace only 1 of 4 RAG-sourced
  candidates turned out to be available). The orchestrator now forces
  fallback mode on from pass 2 onward regardless of the verdict, and
  `react_agent` enforces a per-pass `rag_search` call budget
  (`RAG_BUDGET_BY_PASS`, default unlimited → 1 → 0): pass 1 still searches RAG
  freely, pass 2 treats `tmdb_fallback_search` as primary with at most one
  RAG call left for a genuinely new angle, and pass 3+ disables `rag_search`
  outright. The system prompt's retrieval-tool guidance is now generated per
  pass to match.
  *Why:* the RAG index is the weakest link in the pipeline (mediocre
  taste-matching, zero availability awareness), so later passes — which exist
  specifically to recover from availability/taste rejections — should lean on
  the tool that's actually availability-aware instead of re-querying the same
  weak source.
- **Taste extraction now separates hard subject-matter requirements from soft
  mood/tone, and constrains genre names.** Added a `themes` field to
  `UserPreferences` (e.g. `["mixed-race couple"]`) for concrete plot/subject
  requirements the user names, distinct from `mood`'s tone/vibe text. The
  Taste Extractor prompt now also enumerates the exact 19 TMDB genre names
  instead of leaving `genres` free-form (casual terms like "rom-com" or
  "sci-fi" previously risked producing a genre string `rag_search`'s exact
  filter would never match).
  *Why:* setup for the Reflection fix below — a hard/soft constraint
  hierarchy needs a structured place to put the hard constraints instead of
  burying them in free text.
- **Reflection can no longer trade away a defining constraint for "niche."**
  In a real trace, Reflection rejected the *only* candidate matching the
  user's explicit "mixed-race couple" ask — solely because it judged the
  movie too mainstream — leaving the final answer with two recommendations
  that didn't feature a mixed-race couple at all. Root cause: TMDB's
  `popularity` field (passed to Reflection as the only mainstream-ness
  signal) is a unitless, constantly-rescaled trending metric, not a measure
  of fame, so the model's calibration of it was arbitrary. Fixed by (1)
  surfacing `vote_count` from TMDB (a real proxy: roughly <500 = niche,
  >5000 = mainstream) instead of `popularity`, and (2) adding an explicit
  constraint hierarchy to both the Reflection and ReAct prompts: anything in
  `themes`/`genres`/`similar_to`/`exclude` is a HARD constraint a candidate
  must satisfy outright, while `mood`'s tone/vibe words are SOFT preferences
  that must never override a HARD-constraint match. Reflection is now told
  explicitly: never reject the only candidate satisfying a hard constraint
  for missing a soft one — approve it and ask (via critique) for more,
  nicher options that *also* satisfy the same hard constraint.
  *Why:* the whole point of the mixed-race-couple ask was the hard
  constraint; "niche" was a nice-to-have on top of it, but the agent had no
  way to know which preference was allowed to lose.
- **`tmdb_fallback_search`'s keyword filter no longer dilutes the defining
  constraint with an OR.** The old single `keywords` argument was always
  pipe-joined (`a|b|c` = TMDB's OR), so a query like `["interracial couple",
  "romantic comedy", "sweet romance", "quirky", "small town"]` matched *any
  one* term — which is how a movie with zero mixed-race-couple content
  (matched only on "romantic comedy") outranked the actual match in a real
  trace. Split the argument into `keywords_all` (AND-joined — the hard,
  non-negotiable terms) and `keywords_any` (OR-joined flavor terms, only used
  when `keywords_all` is empty, since TMDB's `with_keywords` can't mix AND
  and OR in one query). Also fixed `_resolve_keyword_ids` to prefer an exact
  (case-insensitive) name match from TMDB's keyword search instead of
  blindly taking the first hit.
  *Why:* for a request built around one defining constraint, OR-only
  filtering means that constraint is just one vote among several — exactly
  the failure mode that put a movie with no mixed-race couple at the top of
  the results.
- **`rag_search`/`tmdb_fallback_search` now deterministically drop
  already-approved/excluded movies from their own results**, instead of
  relying on the model to notice the "Already approved" / "Excluded" lists in
  the prompt. A real trace showed pass 3 re-retrieving three movies
  (`Playing by Heart`, `Meet Cute`, `You People`) that were already excluded
  in pass 2 — wasted tool calls and wasted verification. `_run_tool` now
  filters both tools' `results` against the current pass's known tmdb_ids
  and reports how many were filtered (`filtered_out`), so the model still
  sees why its result count shrank.
  *Why:* prompt instructions are a strong nudge, not a guarantee — this is
  the kind of check that's cheap and exact to do in code and shouldn't be
  left to the model's attention.
- **`tmdb_fallback_search` no longer silently claims availability for an
  unmapped country.** If `resolve_region_code` can't map the given country
  (anything outside the small `COUNTRY_TO_REGION_CODE` table), the tool used
  to just omit `watch_region`/`with_watch_providers` from the TMDB query and
  still stamp every result `"available": True` — an unfiltered global result
  set presented as confirmed-available. It now returns an explicit
  `unknown_country` error with zero results instead. Also stopped trusting
  the model to copy `country`/`platforms` into every call — `react_agent`
  now injects them server-side from the parsed preferences (they're removed
  from the schema's `required` list accordingly), so a mismatched or garbled
  copy can no longer produce wrong availability results.
  *Why:* silently-wrong "available" is worse than an obvious error — the
  whole point of this agent is never claiming a movie is available without a
  real, region-aware lookup.
- **The RAG-outage retry no longer swallows unrelated bugs.** `orchestrator._run_react_with_fallback`'s
  bare `except Exception` was meant to catch "Pinecone is unreachable" — but
  every tool-execution error is already caught and turned into feedback
  *inside* `react_agent`'s own tool loop, so nothing RAG-related ever
  actually reached this handler. In practice it was catching things like the
  "too many invalid tool calls in a row" `RuntimeError` and any LLM
  connection/auth error, silently flipping on fallback mode and re-running
  the *entire* ReAct pass instead of surfacing the real bug. Added a
  dedicated `RagUnavailable` exception, raised from `rag_search.execute` only
  for an embedding-call or vector-backend failure and explicitly re-raised
  (not swallowed) out of `react_agent`'s tool loop; the orchestrator now
  catches exactly that type.
  *Why:* a blanket `except Exception` around a whole agent pass should be
  reserved for the one failure it's actually designed to recover from —
  otherwise it doubles the cost of every other bug and hides what actually
  went wrong.
- **`TMDBClient` reuses one `httpx.AsyncClient` instead of opening a new one
  per request.** `verify_recommendation` makes 3 TMDB calls per candidate (a
  5-candidate draft = 15 calls); each was paying a fresh TCP+TLS handshake
  before this fix. The client is now created lazily on first use and reused
  for the process's lifetime.
  *Why:* pure wasted latency with no upside — this is the single biggest
  win available outside of the pass-count/retry reductions above.
- **`rag_search`'s `min_score` scale is now documented and defended against
  mix-ups.** The schema called it "minimum IMDB rating" while the indexed
  `score` values are actually 0-100 (e.g. 58, 66 in a real trace) — a model
  forwarding `preferences.min_rating` (0-10, e.g. 7.0) as-is would silently
  produce a no-op filter. The schema description now states the 0-100 scale
  explicitly, and `_build_filter` defensively multiplies any value ≤ 10 by 10.
  *Why:* a silent no-op filter is a worse failure mode than a slightly
  redundant defensive check — the model has no way to notice its rating
  filter did nothing.

---

## Trace-driven hardening (2026-07-13)

A real conversation trace ("cute niche rom-com on Netflix Israel with a
mixed-race couple") surfaced a final answer where 2 of 3 recommendations
didn't actually feature a mixed-race couple. Fixes below trace back to that
run plus an independent audit of the same code paths.

- **`tmdb_fallback_search`'s keyword resolution is now a real fuzzy lookup
  against `/search/keyword`, and a failed hard-keyword match is never
  silently swapped for the soft `keywords_any` filter.** Previously
  `_resolve_keyword_ids` only tried an exact (case-insensitive) name match on
  the whole phrase; "mixed-race couple" has no such TMDB keyword (the real
  one is "interracial relationship"), so `keyword_ids_all` came back empty
  and the code fell through to OR-ing the *soft* `keywords_any` terms
  instead ("cute", "rom-com", …) — exactly how a snowman rom-com with zero
  interracial content passed as a verified match in the trace. Resolution
  now tries, per term: exact match → substring match either way → the same
  two checks against each significant sub-word of the phrase (via more
  `/search/keyword` calls). If a hard term still can't be resolved, the tool
  result now reports it explicitly (`unresolved_keywords_all`,
  `keyword_filter_applied`, `warning`) instead of quietly degrading to a
  softer filter, so the retrieval agent knows to retry with a different term
  rather than trust an unfiltered result set.
  *Why:* a silently-unapplied hard filter is indistinguishable from a
  correctly-applied one in the tool's output — the only fix is to make the
  degradation visible.
- **Reflection can no longer bank a hard-constraint failure as "approved" —
  and every candidate must land in exactly one bucket.** In the trace,
  Reflection's own critique said a candidate "does not match the
  mixed-race-couple theme at all," yet the same verdict listed it in
  `approved_ids` as a buffer while asking for better options — and approved
  movies are never re-litigated, so it rode all the way to the final answer.
  The Reflection prompt now states plainly that a HARD-constraint failure
  goes in `rejected`, never `approved_ids`, even as filler; and that every
  candidate must appear in exactly one of the two lists. As a backstop,
  `orchestrator._apply_verdict` now treats any candidate the model leaves in
  neither list (or puts in both) as rejected, instead of letting it vanish
  from cross-pass state entirely.
  *Why:* prompt instructions are a strong nudge, not a guarantee — same
  reasoning as the existing dedup/availability backstops above.
- **Reflection now judges themes semantically instead of by exact wording,
  and the ReAct prompt no longer invites non-matching "buffer" candidates.**
  The trace also showed Reflection downgrading the one correct match because
  its TMDB keyword ("interracial relationship") wasn't the user's literal
  phrase ("mixed-race couple") — a false negative, not a stricter read. The
  Reflection prompt now says themes are judged by meaning, not exact
  wording. Separately, `react_agent`'s "a couple extra as buffer is fine"
  instruction was being read literally as license to include movies the
  model *knew* failed the theme; it now explicitly says buffers exist only
  for availability attrition, and must still satisfy every hard constraint.
  On the final pass, Reflection may also now drop a previously-approved
  movie from the response if it clearly violates a hard constraint on
  reinspection, rather than being forced to include every past approval
  no matter what.
- **Any real access now counts as "available," consistently.** Previously
  `tmdb_fallback_search` restricted discovery to `flatrate|free|ads`
  (subscription-style) while `verify_recommendation` counted `rent`/`buy`
  too — two different notions of "available" in the same pipeline. Per
  product decision, if the user can access the movie there at all (rent, buy,
  or subscription), it counts. `tmdb_fallback_search` now also queries
  `flatrate|free|ads|rent|buy`; this corrects the "Cleaner monetization
  filter" bullet in the entry above, which had gone the other way.
- **Streaming platform names are now resolved case-insensitively, with
  aliases for common alternate spellings — and an unresolvable platform now
  fails loudly instead of silently dropping the filter.** `resolve_provider_ids`
  was a case-sensitive exact-match lookup; the app's own example prompt
  suggests "Disney+", which isn't a key in `PLATFORM_TO_PROVIDER_ID`
  ("Disney Plus" is) — so following the app's own suggestion silently
  produced an unfiltered-by-platform result still stamped `"available":
  true`. Added `PLATFORM_NAME_ALIASES` (Disney+, Apple TV+, Prime Video, HBO
  Max, Paramount+, Peacock, …) and a case-insensitive index, plus
  `resolve_provider_ids_verbose` so `tmdb_fallback_search` can now return an
  explicit `unknown_platform` error (mirroring the existing `unknown_country`
  guard) instead of pretending it filtered.
  *Why:* same principle as the country guard above — silently-wrong
  "available" is worse than a clear error.
- **API error responses no longer discard the collected trace.** `POST
  /api/execute`'s error branches returned `"steps": []` even though `steps`
  held everything gathered before the failure (and was already being sent to
  Supabase). Both branches now return the real `steps` list, so a failed run
  is debuggable from the GUI's trace viewer instead of blank.
- **`remaining_needed` no longer over-asks once most candidates are already
  approved.** It previously floored at `MIN_CANDIDATES` (3) even with e.g. 4
  of 5 slots already filled, prompting an unnecessary search-and-verify round
  for 1 missing slot. Now floors at 1; `MIN_CANDIDATES` remains only
  Reflection's approve-threshold.