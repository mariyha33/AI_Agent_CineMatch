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

---

## Run locally

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env        # then fill in real keys (see below)
uvicorn api.index:app --reload
```

Open <http://localhost:8000/docs> for interactive Swagger, or:

```bash
curl -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Slow-burn crime like Sicario, on Netflix in Israel"}'
```

> Requires Python 3.12 (matches the Vercel runtime).

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

1. Push this repo to GitHub and import it into Vercel.
2. Add all environment variables above in **Project → Settings → Environment
   Variables**.
3. Deploy. `vercel.json` routes `/api/*` to the Python function with a 300 s max
   duration.
4. Verify: `GET /api/team_info`, `GET /api/agent_info`,
   `GET /api/model_architecture`, and a `POST /api/execute`.

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
6. **Build the frontend** — a minimal page at `/` with a prompt box, a "Run
   Agent" button calling `POST /api/execute`, and a display of `response` + the
   full `steps` trace. Deferred to a later phase; no auth guards.

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
public/architecture.png   (you provide this)
```
