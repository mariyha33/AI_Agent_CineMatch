# CineMatch

CineMatch is an AI agent that recommends **movies** matched to your taste and
**verified as actually watchable** on the streaming service you have, in the
country you're in. It never tells you a film is "on Netflix" without a real,
live availability check first. Movies only — no TV.

---

## 1. What it does, in plain terms

You tell CineMatch what you're in the mood for — in natural language or as a
short structured brief — for example:

> *"Slow-burn crime like Sicario, on Netflix in Israel. I've already seen
> Prisoners."*

CineMatch then:

1. **Understands your taste.** It reads your message and pulls out the vibe
   (mood/tone), any hard requirements (genres, specific subject matter like
   "heist" or "mixed-race couple"), movies you love, movies to avoid, your
   country, your streaming platforms, year range, and minimum rating.
2. **Finds candidate movies.** It searches a vector index of ~9,000 IMDB movies
   for titles that semantically match your taste, and/or queries TMDB's
   discovery API for more obscure or precisely-filtered picks.
3. **Verifies availability for real.** For every candidate it calls the live
   TMDB API to confirm the movie is actually streamable/rentable/buyable on
   *your* platform in *your* country. Anything that isn't available is thrown
   out.
4. **Judges quality and taste fit.** A critic step decides whether the
   surviving movies genuinely match what you asked for — rejecting mismatches
   and, if needed, sending the search back for another round.
5. **Answers you.** It returns a short, formatted list of recommendations, each
   with a one-line reason and the exact platform + country where you can watch
   it. If your request is too vague (and you're using the chat UI), it asks a
   single clarifying question instead.

The two problems it exists to solve — the two things a plain LLM gets wrong for
movie recommendations — are **genuine personalization** (not just "the most
popular title") and **real, country- and platform-specific availability**.

### What it can't do

- No TV shows (movies only).
- It never asserts availability without a live TMDB lookup.
- No user accounts, watch history, or personal data beyond what you type.
- It can't honor runtime/duration filters from the vector index alone — those are only possible through TMDB discovery.

---

## 2. Architecture, tools, and logic

### The big picture

```
User prompt
   │
   ▼
┌──────────────────┐  Stage 0
│ TasteExtractor   │  1 LLM call → structured UserPreferences
└──────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  ReAct ⇄ Reflection loop  (up to MAX_PASSES = 3 passes)      │
│                                                              │
│  ┌──────────────┐  Stage 1                                   │
│  │ ReActAgent   │  agentic tool loop → draft candidate list  │
│  │  tools:      │                                            │
│  │   rag_search              (semantic taste search)         │
│  │   tmdb_fallback_search    (structured TMDB discovery)     │
│  │   ask_user_clarification  (interactive mode only)         │
│  └──────────────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────┐   deterministic, no LLM        │
│  │ verify_recommendation    │   live TMDB availability check │
│  │ (Orchestrator)           │   → drop unavailable movies    │
│  └──────────────────────────┘                                │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐  Stage 2                                   │
│  │ ReflectionAgent │  1 LLM call → per-candidate verdict     │
│  │              │   ├─ approve → compose final answer → DONE │
│  │              │   ├─ reject  → feed critique back to ReAct │
│  │              │   └─ clarify → ask the user (interactive)  │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

Module names are identical across the code, the `steps` trace returned by the
API, and (by design) the architecture diagram: `TasteExtractor`, `ReActAgent`,
`ReActAgent/rag_search`, `ReActAgent/tmdb_fallback_search`,
`ReActAgent/ask_user_clarification`, `Orchestrator/verify_recommendation`,
`ReflectionAgent`.

### Stage 0 — TasteExtractor ([agent/taste_extractor.py](agent/taste_extractor.py))

A single LLM call turns free-text (or conversation history) into a structured
`UserPreferences` object: `mood`, `genres` (constrained to TMDB's 19 canonical
genre names), `themes` (hard subject-matter requirements), `similar_to`,
`exclude`, `country`, `platforms`, `year_min/max`, `min_rating`. If the incoming
prompt is *already* a JSON object matching the schema, the LLM call is **skipped
entirely** (a zero-cost structured-input bypass).

Key design point: **hard vs. soft constraints.** `themes`/`genres`/`similar_to`/
`exclude` and year/rating are HARD (a candidate must satisfy them); `mood` is
SOFT (tone/vibe to lean into, never a pass/fail gate). This distinction drives
the whole rest of the pipeline.

### Stage 1 — ReActAgent ([agent/react_agent.py](agent/react_agent.py))

A true agentic loop using the LLM's native function-calling. The model decides
which tool to call, with what arguments, and when to stop. When it stops it
emits a JSON draft of candidate movies (each with a `tmdb_id`). Tool calls
issued in the same turn run **concurrently** (`asyncio.gather`).

Its tools:

| Tool | File | What it does |
|---|---|---|
| `rag_search` | [agent/tools/rag_search.py](agent/tools/rag_search.py) | Semantic vector search over the ~9k-movie IMDB catalog. Returns `tmdb_id` + metadata. **No availability signal.** Backed by Pinecone. |
| `tmdb_fallback_search` | [agent/tools/tmdb_fallback_search.py](agent/tools/tmdb_fallback_search.py) | Structured discovery via TMDB `/discover/movie`. Results are **already availability-filtered**. Supports vote-count/runtime/language filters, AND-joined hard keywords (`keywords_all`) vs OR-joined flavor keywords (`keywords_any`), and quality-based sorting. |
| `ask_user_clarification` | [agent/tools/ask_user_clarification.py](agent/tools/ask_user_clarification.py) | Signals that the request is too vague. Only offered in interactive mode. |

**Retrieval policy shifts across passes.** The RAG index has no availability
data (in practice a low fraction of RAG hits turn out to be streamable), so the
orchestrator forces `tmdb_fallback_search` on from pass 2, and a per-pass
`rag_search` budget (`RAG_BUDGET_BY_PASS`, default `-1,1,0`) limits RAG calls:
unlimited on pass 1, one "new angle" on pass 2, none from pass 3 on.

### Deterministic availability check — verify_recommendation ([agent/tools/verify_recommendation.py](agent/tools/verify_recommendation.py))

This is a tool, but it is **not** called by an LLM. The orchestrator invokes it
directly as a plain async function right after each ReAct draft. For each
candidate (in parallel) it fetches TMDB movie detail + watch providers +
keywords, and decides `pass`/`fail` on availability. Any real access tier —
`flatrate`, `free`, `ads`, `rent`, or `buy` — counts as "available." Only
availability-confirmed movies reach Reflection. Doing this in code (rather than
as an LLM tool call) removes an entire LLM round-trip per pass — availability is
a binary lookup, not a judgment call.

### Stage 2 — ReflectionAgent ([agent/reflection_agent.py](agent/reflection_agent.py))

A single LLM call (no tool loop — availability is already settled). It receives
the availability-confirmed candidates and judges **taste and novelty only**,
returning a per-candidate verdict:

- `approved_ids` — kept this pass (banked toward the final answer).
- `rejected` — with a reason each.
- `decision` — overall `approve` / `reject` / `clarify`.
- `final_response` — the user-facing text, composed by Reflection itself on
  approve (there is no separate composer stage).
- `critique` — actionable feedback fed back to ReActAgent on reject.

It enforces the hard/soft hierarchy: it must never reject the *only* candidate
satisfying a hard constraint just for being more mainstream than the soft
"niche" wish. It judges niche-ness by TMDB `vote_count` (a real fame proxy),
not the volatile `popularity` field.

### The orchestrator loop ([agent/orchestrator.py](agent/orchestrator.py))

Plain Python, no LLM routing. It runs Stage 0 once, then loops Stage 1 → verify
→ Stage 2 up to `MAX_PASSES` times, carrying **cross-pass memory**:

- `approved` movies accumulate and are never re-searched.
- `excluded` movies (unavailable or taste-rejected) are listed back to ReAct so
  it looks for *genuinely new* titles instead of re-surfacing rejects. Both
  tools also drop already-known movies from their own results in code, not just
  by prompt instruction.
- The loop exits as soon as enough movies are approved.
- On the final pass, "reject" is disallowed — Reflection must compose the best
  possible answer from whatever was approved, so the user always gets a real
  response.

### Cross-turn memory: `SessionState`

The server is a stateless Vercel function, so anything that must survive
*between conversation turns* (movies already ruled out, movies already shown) is
returned to the client as `state` and echoed back as `prior_state` on the next
call. This stops a "Try again" turn from re-drafting and re-rejecting the exact
movies the previous turn already handled. The chat UI round-trips this
automatically. (As a belt-and-braces fallback, TasteExtractor also scans the
conversation history for already-recommended titles.)

### Interactive vs. non-interactive mode

Whether the agent may ask a question is inferred from `conversation_history`:

- **Present** (a list, even empty `[]`) → **interactive**. The GUI sends this,
  so `ask_user_clarification` is offered and a reply may be a follow-up
  question.
- **Omitted** (`null`) → **non-interactive**. Automated evaluators can't answer
  questions, so clarification is withheld and the pipeline **always** returns
  recommendations, making sensible assumptions for anything missing.

### Robustness details

- **Retry-on-transient** for LLM calls (timeout/5xx, one retry).
- **TMDB client reuses one `httpx.AsyncClient`** across all calls (avoids a
  fresh TLS handshake per request — ~15 calls for a 5-candidate draft).
- **`RagUnavailable`** is a dedicated exception so a genuine RAG/Pinecone outage
  triggers a single fallback retry, while every *other* bug surfaces
  immediately instead of being silently swallowed.
- **Loud failures over silent wrong answers**: an unmappable country or platform
  returns an explicit error rather than an unfiltered result set falsely stamped
  "available."
- **Invalid-tool-call cap** aborts a runaway ReAct loop after too many bad calls
  in a row.

### The API ([api/index.py](api/index.py))

A single FastAPI app deployed as one Vercel Python serverless function.

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | Serves the chat UI (`public/index.html`) |
| GET  | `/api/team_info` | Student/team details |
| GET  | `/api/agent_info` | Description, purpose, prompt template, worked examples + step traces |
| GET  | `/api/model_architecture` | Architecture diagram PNG (404 until you add it) |
| POST | `/api/execute` | Main entry point — returns `response` + full `steps` trace + `state` |

`POST /api/execute` request body:

```json
{
  "prompt": "Something like Sicario on Netflix in Israel",
  "conversation_history": [],
  "prior_state": null,
  "session_id": null
}
```

`conversation_history`, `prior_state`, and `session_id` are all optional.
Response shape:

```json
{ "status": "ok", "error": null, "response": "…", "steps": [ … ], "state": { … } }
```

Every LLM call and every tool execution appends one object to `steps`
(`{module, prompt: {system_prompt, user_prompt}, response}`), so the whole run
is fully traceable. Errors return `status: "error"` with the trace collected so
far preserved (not blanked).

---

## 3. Running it locally

### Prerequisites

- Python 3.12 (matches the Vercel runtime).
- An LLMod.ai API key (and the model IDs), plus a TMDB API key.

### Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate            # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` in the project root with at least:

```
LLMOD_API_KEY=...
LLMOD_BASE_URL=https://api.llmod.ai/v1
LLMOD_MODEL=MB5R2CF-azure/gpt-5.4-mini
LLMOD_EMBEDDING_MODEL=MB5R2CF-azure/text-embedding-3-small
TMDB_API_KEY=...
```

Optional (the app runs fully without these):

```
PINECONE_API_KEY=...            # only if RAG_BACKEND=pinecone
PINECONE_INDEX_NAME=cinematch-movies
SUPABASE_URL=...                # run-logging is best-effort
SUPABASE_KEY=...
RAG_BACKEND=local               # "local" (default) or "pinecone"
RAG_DOCUMENTS_PATH=data/processed/rag_documents.jsonl
```

Tunable knobs (all have sensible defaults — see [agent/config.py](agent/config.py)):
`MAX_PASSES`, `MAX_TOOL_CALLS`, `RAG_TOP_K`, `MIN_CANDIDATES`, `MAX_CANDIDATES`,
`RAG_BUDGET_BY_PASS`, `MAX_CONSECUTIVE_INVALID_TOOL_CALLS`.

### Build the RAG index

`rag_search` reads from either a local JSONL file (`RAG_BACKEND=local`, default —
no Pinecone needed) or a Pinecone index. Both are built the same way, from
`data/processed/canonical_movies.csv` only (never `unmatched_movies.csv` — those
rows have no verified `tmdb_id`).

```bash
# Smoke test — no API key, no cost, exercises the whole pipeline:
python scripts/build_rag_index.py --limit 20 --mock-embeddings

# Real embeddings for a small sample (uses LLMOD_API_KEY):
python scripts/build_rag_index.py --limit 20

# Full build (all ~9,160 canonical movies):
python scripts/build_rag_index.py

# Also push vectors to Pinecone (ids are deterministic: movie-{tmdb_id}):
python scripts/build_rag_index.py --upsert-pinecone
```

This writes `data/processed/rag_documents.jsonl`, one JSON object per movie
(`tmdb_id, title, year, genres, overview, score, runtime, embedding`).

### Run the server

```bash
uvicorn api.index:app --reload
```

Then:

- Open <http://localhost:8000/> for the chat UI.
- Open <http://localhost:8000/docs> for interactive Swagger.
- Or call the API directly:

```bash
curl -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Slow-burn crime like Sicario, on Netflix in Israel"}'
```

### Using the chat UI

The UI ([public/index.html](public/index.html)) is a single self-contained file
(inline CSS + vanilla JS, no build step, no auth). It:

- Sends every turn to `POST /api/execute` with `conversation_history` (and the
  round-tripped `prior_state`), keeping the pipeline in interactive mode.
- Renders the reply as markdown, with a collapsible **execution trace** showing
  each step's module, prompts, and response.
- After recommendations, shows **Satisfied** (ends the chat client-side, no API
  call) and **Try again** (type what was wrong; reruns with that as feedback).
- When the agent asks a clarifying question, just leaves the input open for your
  answer, which is appended to the history.

### Deploying to Vercel

`vercel.json` routes `/api/*` to the Python function (300 s max duration) and
serves `public/` as static assets. Import the repo in Vercel (framework preset
**Other** — no build command), add the same environment variables in project
settings, and deploy. Every push to the connected branch redeploys
automatically.

---

## 4. What each file does

### Entry point & config

| File | Role |
|---|---|
| [api/index.py](api/index.py) | FastAPI app; all four `/api/*` routes + the `/` UI route; best-effort Supabase logging. |
| [agent/config.py](agent/config.py) | Loads all env vars and tunable parameters; required vars validated lazily (import stays cheap for mock workflows). |
| [vercel.json](vercel.json) | Vercel routing + 300 s function duration. |
| [requirements.txt](requirements.txt) | Python dependencies (fastapi, uvicorn, openai, pinecone, supabase, httpx, pydantic, python-dotenv). |

### The pipeline (`agent/`)

| File | Role |
|---|---|
| [agent/orchestrator.py](agent/orchestrator.py) | The deterministic loop controller — the brain. Runs Stage 0, then the bounded ReAct⇄verify⇄Reflection loop; manages cross-pass and cross-turn state; composes the final answer / best-effort fallback. |
| [agent/taste_extractor.py](agent/taste_extractor.py) | Stage 0 — parses the request into `UserPreferences` (with a zero-LLM structured-input bypass). |
| [agent/react_agent.py](agent/react_agent.py) | Stage 1 — the agentic retrieval loop (native function-calling, parallel tool calls, per-pass RAG budget). |
| [agent/reflection_agent.py](agent/reflection_agent.py) | Stage 2 — single-LLM-call taste/novelty critic; composes the final response on approve. |
| [agent/prompts.py](agent/prompts.py) | Every system prompt + the user-facing prompt template; per-pass prompt assembly for ReAct and Reflection. |
| [agent/models.py](agent/models.py) | Pydantic models for every data shape (`UserPreferences`, `Candidate`, `ReactContext`, `ReflectionVerdict`, `SessionState`, request/response, etc.). |
| [agent/agent_info.py](agent/agent_info.py) | Static metadata served by `/api/agent_info` — description, purpose, template, worked examples with real step traces. |
| [agent/parsing.py](agent/parsing.py) | Helpers to coerce LLM text into JSON (strips markdown fences, extracts the first `{…}` span). |
| [agent/steps.py](agent/steps.py) | The one helper that appends a trace step in the spec's fixed schema. |
| [agent/tmdb_mappings.py](agent/tmdb_mappings.py) | Lookup tables + resolvers: platform name → TMDB provider ID (with aliases, case-insensitive), country → ISO region, TMDB genre ID → name. |

### Tools (`agent/tools/`)

| File | Role |
|---|---|
| [agent/tools/registry.py](agent/tools/registry.py) | Maps tool name → (OpenAI schema, async executor); exposes `schemas_for(...)` to hand the LLM only the tools allowed this pass. |
| [agent/tools/rag_search.py](agent/tools/rag_search.py) | Semantic vector search tool; defines `RagUnavailable`. |
| [agent/tools/tmdb_fallback_search.py](agent/tools/tmdb_fallback_search.py) | Structured, availability-filtered TMDB discovery; fuzzy keyword resolution with loud reporting of unresolved hard keywords. |
| [agent/tools/verify_recommendation.py](agent/tools/verify_recommendation.py) | Deterministic live availability check (called by the orchestrator, not an LLM); defines `TMDBUnavailable`. |
| [agent/tools/ask_user_clarification.py](agent/tools/ask_user_clarification.py) | Emits the "ask the user" signal (no external call). |

### Clients (`agent/clients/`)

| File | Role |
|---|---|
| [agent/clients/llm_client.py](agent/clients/llm_client.py) | Async wrapper over the OpenAI SDK pointed at LLMod.ai; `chat()` + `embed()`; retry-once; lazy client construction. |
| [agent/clients/tmdb_client.py](agent/clients/tmdb_client.py) | Async TMDB helper (httpx) with a reused connection; movie detail, watch providers, keywords, discover, keyword search. |
| [agent/clients/local_rag_client.py](agent/clients/local_rag_client.py) | Zero-dependency local JSONL vector search (cosine similarity + filters) — the default RAG backend. |
| [agent/clients/pinecone_client.py](agent/clients/pinecone_client.py) | Pinecone query/upsert helper; lazy connection so the app imports cleanly without Pinecone. |
| [agent/clients/supabase_client.py](agent/clients/supabase_client.py) | Supabase run-logging + conversation persistence (best-effort). |

### Data & scripts

| File | Role |
|---|---|
| [scripts/build_canonical_movies.py](scripts/build_canonical_movies.py) | **Phase 1** data pipeline: cleans the raw IMDB CSV and resolves each movie to a verified TMDB id (fuzzy title/year/language/overview scoring), splitting rows into `canonical_movies.csv` vs `unmatched_movies.csv`. No LLM. |
| [scripts/build_rag_index.py](scripts/build_rag_index.py) | **Phase 2**: embeds each canonical movie and writes `rag_documents.jsonl` (and optionally upserts to Pinecone). Supports `--limit`, `--mock-embeddings`, `--overwrite`. |
| [data/imdb_movies.csv](data/imdb_movies.csv) | Raw IMDB export (pipeline input). |
| [data/processed/canonical_movies.csv](data/processed/canonical_movies.csv) | ~9,160 TMDB-verified movies — the sole source for the RAG index. |
| [data/processed/unmatched_movies.csv](data/processed/unmatched_movies.csv) | Rows dropped during cleaning or with no confident TMDB match (never recommended). |
| [data/processed/canonical_build_report.json](data/processed/canonical_build_report.json) | Phase-1 build stats (92.2% match rate). |

### Frontend

| File | Role |
|---|---|
| [public/index.html](public/index.html) | The entire chat UI — one self-contained file, served at `/`. |
| `public/architecture.png` | Architecture diagram PNG — **you must add this** (see below). |

---

## 5. What's left to do

Measured against [markdown_files/Project instructions.md](markdown_files/Project%20instructions.md),
the core is complete: all four required endpoints exist, the GUI is built with
no auth guards and displays both the response and the full step trace, the
pipeline is optimized (deterministic control flow, minimal LLM calls, parallel
tool/verify calls), and it's Vercel-ready. Remaining gaps:

1. **Architecture diagram (required).** `GET /api/model_architecture` returns a
   404 until you drop a PNG at `public/architecture.png`. The instructions
   require it to be clear and to use **exactly** the module names the code and
   `steps` trace use (see the list in §2). — *Not yet present.*

2. **Real team info (required).** [api/index.py](api/index.py) `/api/team_info`
   still returns placeholders (`"PLACEHOLDER"`, `Student A/B/C`,
   `a@example.com`). Replace with real names, emails, and the
   `{batch#}_{order#}` group number from the presentation list.

3. **Improve and test single-turn prompts.** The current prompt template is a first-pass draft. It works
   but is not yet polished for clarity, brevity, and robustness. The worked
   examples in [agent/agent_info.py](agent/agent_info.py) are a good starting
   point for testing.
