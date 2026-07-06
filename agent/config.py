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
LLMOD_API_KEY = _require("LLMOD_API_KEY")
LLMOD_BASE_URL = _optional("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
LLMOD_MODEL = _optional("LLMOD_MODEL", "MB5R2CF-azure/gpt-5.4-mini")
LLMOD_EMBEDDING_MODEL = _optional(
    "LLMOD_EMBEDDING_MODEL", "MB5R2CF-azure/text-embedding-3-small"
)

# --- TMDB --------------------------------------------------------------------
TMDB_API_KEY = _require("TMDB_API_KEY")
TMDB_BASE_URL = _optional("TMDB_BASE_URL", "https://api.themoviedb.org/3")

# --- Pinecone ----------------------------------------------------------------
PINECONE_API_KEY = _require("PINECONE_API_KEY")
PINECONE_INDEX_NAME = _optional("PINECONE_INDEX_NAME", "cinematch-movies")

# --- Supabase ----------------------------------------------------------------
SUPABASE_URL = _require("SUPABASE_URL")
SUPABASE_KEY = _require("SUPABASE_KEY")


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
