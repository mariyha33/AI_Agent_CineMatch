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
{ "prompt": "Something like Sicario on Netflix in Israel", "conversation_history": [] }
```

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
`MAX_CANDIDATES`. See `.env.example`.

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
