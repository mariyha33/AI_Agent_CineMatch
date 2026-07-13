"""Central configuration for CineMatch.

All environment variables and tunable parameters are loaded here. Required
variables raise a clear error at import time if missing; tunables fall back to
sensible defaults.
"""
from __future__ import annotations

import os

try:
    # Optional: load a local .env during development. In production (Vercel)
    # environment variables are injected directly, so this is a no-op.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and fill it in (see README)."
        )
    return value


def _optional(name: str, default: str) -> str:
    return os.environ.get(name) or default


# --- LLM (LLMod.ai, OpenAI-compatible) ---------------------------------------
# Optional at import time: only required when a real chat()/embed() call is
# made (see agent/clients/llm_client.py). This lets mock-embedding workflows
# (e.g. scripts/build_rag_index.py --mock-embeddings) import agent.config with
# zero external credentials.
LLMOD_API_KEY = _optional("LLMOD_API_KEY", "")
LLMOD_BASE_URL = _optional("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
LLMOD_MODEL = _optional("LLMOD_MODEL", "MB5R2CF-azure/gpt-5.4-mini")
LLMOD_EMBEDDING_MODEL = _optional(
    "LLMOD_EMBEDDING_MODEL", "MB5R2CF-azure/text-embedding-3-small"
)

# --- TMDB --------------------------------------------------------------------
# Optional at import time: only required when a real TMDB API call is made
# (see agent/clients/tmdb_client.py). Local/mock RAG commands (e.g.
# scripts/build_rag_index.py --mock-embeddings) never touch TMDB.
TMDB_API_KEY = _optional("TMDB_API_KEY", "")
TMDB_BASE_URL = _optional("TMDB_BASE_URL", "https://api.themoviedb.org/3")

# --- Pinecone ------------------------------------------------------------
# Optional at import time: only required when RAG_BACKEND="pinecone" (or when
# scripts/build_rag_index.py is run with --upsert-pinecone). Local dev/tests
# must be able to import this module with no Pinecone account at all.
PINECONE_API_KEY = _optional("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = _optional("PINECONE_INDEX_NAME", "cinematch-movies")

# --- Supabase --------------------------------------------------------------
# Optional at import time: run logging is best-effort (see api/index.py), so
# the app must work even before Supabase is configured.
SUPABASE_URL = _optional("SUPABASE_URL", "")
SUPABASE_KEY = _optional("SUPABASE_KEY", "")

# --- RAG backend selection ----------------------------------------------------
# "local"    -> JSONL fallback (agent/clients/local_rag_client.py), no external
#               vector DB required. Default so the project runs with zero
#               Pinecone setup.
# "pinecone" -> use the pre-populated Pinecone index via pinecone_client.
RAG_BACKEND = _optional("RAG_BACKEND", "local").strip().lower()
RAG_DOCUMENTS_PATH = _optional(
    "RAG_DOCUMENTS_PATH", "data/processed/rag_documents.jsonl"
)


# --- Tunable pipeline parameters ---------------------------------------------
MAX_PASSES = int(_optional("MAX_PASSES", "2"))          # ReAct <-> Reflection loop
MAX_TOOL_CALLS = int(_optional("MAX_TOOL_CALLS", "8"))  # per ReAct run
RAG_TOP_K = int(_optional("RAG_TOP_K", "10"))           # Pinecone results
MIN_CANDIDATES = int(_optional("MIN_CANDIDATES", "3"))  # min before Reflection
MAX_CANDIDATES = int(_optional("MAX_CANDIDATES", "5"))  # cap on final recs

# Cap on consecutive invalid tool calls before aborting a ReAct run (§12).
MAX_CONSECUTIVE_INVALID_TOOL_CALLS = int(
    _optional("MAX_CONSECUTIVE_INVALID_TOOL_CALLS", "3")
)
