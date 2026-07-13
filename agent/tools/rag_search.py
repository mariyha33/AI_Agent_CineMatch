"""rag_search — semantic movie search over the RAG-indexed movie catalog.

Backed by either Pinecone (RAG_BACKEND=pinecone) or a local JSONL fallback
(RAG_BACKEND=local, the default — see agent/clients/local_rag_client.py). Both
sources are built exclusively from data/processed/canonical_movies.csv, so
every result already carries a valid tmdb_id — no TMDB lookup happens here.
Availability is verified downstream by the Reflection stage.
"""
from __future__ import annotations

from typing import Any, List, Optional

from agent import config
from agent.clients.llm_client import llm_client
from agent.clients.local_rag_client import local_rag_client
from agent.clients.pinecone_client import pinecone_client

TOOL_NAME = "rag_search"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Semantic search over the CineMatch movie catalog (Pinecone). Use "
            "the mood/vibe and any 'similar to' titles as the query text. "
            "Returns candidate movies with their tmdb_id and metadata. Does not "
            "check availability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "Semantic search text derived from mood/vibe and similar-to titles.",
                },
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional genre filter (matches any listed genre).",
                },
                "year_min": {"type": "integer", "description": "Optional earliest release year."},
                "year_max": {"type": "integer", "description": "Optional latest release year."},
                "min_score": {"type": "number", "description": "Optional minimum IMDB rating."},
                "top_k": {"type": "integer", "description": "Number of results to return."},
            },
            "required": ["query_text"],
        },
    },
}


def _build_filter(
    genres: Optional[List[str]],
    year_min: Optional[int],
    year_max: Optional[int],
    min_score: Optional[float],
) -> dict:
    flt: dict = {}
    if genres:
        flt["genres"] = {"$in": genres}
    year_clause: dict = {}
    if year_min is not None:
        year_clause["$gte"] = year_min
    if year_max is not None:
        year_clause["$lte"] = year_max
    if year_clause:
        flt["year"] = year_clause
    if min_score is not None:
        flt["score"] = {"$gte": min_score}
    return flt


async def execute(args: dict) -> dict:
    query_text: str = args["query_text"]
    genres = args.get("genres")
    year_min = args.get("year_min")
    year_max = args.get("year_max")
    min_score = args.get("min_score")
    top_k = int(args.get("top_k") or config.RAG_TOP_K)

    vector = await llm_client.embed(query_text)
    flt = _build_filter(genres, year_min, year_max, min_score)

    use_pinecone = config.RAG_BACKEND == "pinecone"
    if use_pinecone:
        matches = pinecone_client.query(vector=vector, filter=flt, top_k=top_k)
    else:
        matches = local_rag_client.search(vector=vector, filter=flt, top_k=top_k)

    results: List[dict] = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        result = {
            "title": md.get("title"),
            "year": md.get("year"),
            "tmdb_id": md.get("tmdb_id"),
            "genres": md.get("genres", []),
            "score": md.get("score"),
            "overview": md.get("overview"),
        }
        if use_pinecone:
            result["pinecone_score"] = m.get("score")
        else:
            result["local_score"] = m.get("score")
        results.append(result)
    return {"results": results, "count": len(results)}
