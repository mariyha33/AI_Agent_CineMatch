"""Central registry mapping tool names to their schema + execute function."""
from __future__ import annotations

from typing import Callable, Dict

from agent.tools import (
    ask_user_clarification,
    rag_search,
    tmdb_fallback_search,
    verify_recommendation,
)

# name -> (openai_schema_dict, async execute(args) -> dict)
_TOOLS: Dict[str, tuple] = {
    rag_search.TOOL_NAME: (rag_search.SCHEMA, rag_search.execute),
    tmdb_fallback_search.TOOL_NAME: (
        tmdb_fallback_search.SCHEMA,
        tmdb_fallback_search.execute,
    ),
    verify_recommendation.TOOL_NAME: (
        verify_recommendation.SCHEMA,
        verify_recommendation.execute,
    ),
    ask_user_clarification.TOOL_NAME: (
        ask_user_clarification.SCHEMA,
        ask_user_clarification.execute,
    ),
}


def get_schema(name: str) -> dict:
    return _TOOLS[name][0]


def get_executor(name: str) -> Callable:
    return _TOOLS[name][1]


def exists(name: str) -> bool:
    return name in _TOOLS


def schemas_for(names: list[str]) -> list[dict]:
    return [_TOOLS[n][0] for n in names if n in _TOOLS]
