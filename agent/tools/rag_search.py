"""rag_search — semantic movie search over the RAG-indexed movie catalog.

Backed by either Pinecone (RAG_BACKEND=pinecone) or a local JSONL fallback
(RAG_BACKEND=local, the default — see agent/clients/local_rag_client.py). Both
sources are built exclusively from data/processed/canonical_movies.csv, so
every result already carries a valid tmdb_id — no TMDB lookup happens here.
Availability is verified downstream by the Reflection stage.

Phase 3 — vote-count-aware reranking: raw cosine similarity is blended with a
bounded, log-scaled "vote confidence" term derived from each candidate's
vote_count, so a barely-rated title doesn't accidentally outrank a clearly
established match, without letting popularity dominate. See _vote_confidence
and _rerank below.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

from agent import config
from agent.clients.llm_client import llm_client
from agent.clients.local_rag_client import local_rag_client
from agent.clients.pinecone_client import pinecone_client

TOOL_NAME = "rag_search"


class RagUnavailable(RuntimeError):
    """Raised when the embedding call or the vector backend (Pinecone / local
    JSONL index) itself fails — as opposed to a normal empty-results search.

    Distinct from a generic Exception so the orchestrator can retry the whole
    ReAct pass in fallback mode specifically for this failure, instead of for
    any bug in the agent loop (see agent/orchestrator.py
    _run_react_with_fallback).
    """

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Semantic search over the CineMatch movie catalog (Pinecone). Use "
            "the mood/vibe and any 'similar to' titles as the query text. "
            "Do not include the streaming service and country of streaming service in the query text, as availability is checked later. "
            "Returns candidate movies with their tmdb_id and metadata (including "
            "vote_count, vote_average, director, and keywords when available), "
            "reranked by a gentle blend of semantic similarity and vote-count "
            "confidence — see prioritize_niche below. Does not check availability."
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
                "min_score": {
                    "type": "number",
                    "description": (
                        "Optional minimum rating, on the indexed 0-100 scale "
                        "(e.g. 70 = an IMDB rating of 7.0) — NOT 0-10."
                    ),
                },
                "vote_count_min": {
                    "type": "integer",
                    "description": "Optional minimum TMDB vote_count. Results with an unknown or lower vote_count are excluded exactly (not just deprioritized).",
                },
                "vote_count_max": {
                    "type": "integer",
                    "description": "Optional maximum TMDB vote_count. Results with an unknown or higher vote_count are excluded exactly (not just deprioritized).",
                },
                "prioritize_niche": {
                    "type": "boolean",
                    "description": (
                        "Set true ONLY when the user explicitly asked for a "
                        "niche/obscure/lesser-known/indie movie. When true, a "
                        "low vote_count is not treated as a weakness during "
                        "reranking — otherwise a small vote-count confidence "
                        "boost gently favors better-established matches among "
                        "otherwise similar results."
                    ),
                },
                "top_k": {"type": "integer", "description": "Number of results to return."},
            },
            "required": ["query_text"],
        },
    },
}

# How much weight vote-count confidence gets relative to raw semantic
# similarity when reranking. Similarity stays the dominant signal either
# way — this is small enough to only break near-ties or nudge ordering, never
# to override a clearly better semantic match with a clearly worse one.
VOTE_BLEND_WEIGHT_NORMAL = 0.12
# When the user explicitly asked for niche/obscure/lesser-known/indie movies,
# a low vote_count must not be treated as a weakness at all.
VOTE_BLEND_WEIGHT_NICHE = 0.0
# Vote-count confidence saturates here (roughly "mainstream" territory) — a
# blockbuster with 500k votes gets no more credit than one with 5k, so raw
# popularity can never keep dominating past this point.
VOTE_CONFIDENCE_SATURATION = 5000
# Retrieve a larger candidate pool than top_k before filtering/reranking, so
# reranking has real headroom to reorder within — blending after the backend
# has already truncated to top_k would defeat the purpose.
_RERANK_POOL_MULTIPLIER = 3
_RERANK_POOL_MAX = 50


def _vote_confidence(vote_count: Optional[int]) -> float:
    """Bounded 0..1 confidence signal from vote_count, log-scaled so it grows
    quickly for the first few hundred votes and saturates for popular titles.
    None/0/missing vote_count -> 0.0 (no confidence boost, not a penalty —
    the base similarity score is untouched either way)."""
    if not vote_count or vote_count <= 0:
        return 0.0
    return min(math.log1p(vote_count) / math.log1p(VOTE_CONFIDENCE_SATURATION), 1.0)


def _apply_vote_count_bounds(
    results: List[dict], vote_count_min: Optional[int], vote_count_max: Optional[int]
) -> List[dict]:
    """Exact filter (not a reranking nudge): drop any result outside the
    requested vote_count range. A result with an unknown vote_count is
    excluded whenever a bound was explicitly requested, since it can't be
    confirmed to satisfy it."""
    if vote_count_min is None and vote_count_max is None:
        return results
    out = []
    for r in results:
        vc = r.get("vote_count")
        if vote_count_min is not None and (vc is None or vc < vote_count_min):
            continue
        if vote_count_max is not None and (vc is None or vc > vote_count_max):
            continue
        out.append(r)
    return out


def _rerank(results: List[dict], niche: bool) -> List[dict]:
    """Blend each result's raw similarity score with a bounded vote-count
    confidence term, then re-sort by the blend. Semantic similarity remains
    the dominant term; this only nudges ordering."""
    weight = VOTE_BLEND_WEIGHT_NICHE if niche else VOTE_BLEND_WEIGHT_NORMAL
    if weight <= 0:
        return results

    def _blended_score(r: dict) -> float:
        raw_score = r.get("pinecone_score", r.get("local_score"))
        raw_score = raw_score if raw_score is not None else 0.0
        confidence = _vote_confidence(r.get("vote_count"))
        return raw_score * (1 - weight) + confidence * weight

    return sorted(results, key=_blended_score, reverse=True)


def _build_filter(
    genres: Optional[List[str]],
    year_min: Optional[int],
    year_max: Optional[int],
    min_score: Optional[float],
    vote_count_min: Optional[int] = None,
    vote_count_max: Optional[int] = None,
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
        # Defensive: the index's `score` is 0-100, but a model (or a caller
        # forwarding preferences.min_rating, which is 0-10) may pass a 0-10
        # value — treat anything <= 10 as being on that scale.
        if min_score <= 10:
            min_score *= 10
        flt["score"] = {"$gte": min_score}
    # Pushed into the backend metadata filter (both Pinecone's native query
    # filter and local_rag_client.py's own filter parsing — see there) so
    # out-of-range candidates are excluded BEFORE top_k truncation, not just
    # after. _apply_vote_count_bounds (below) still re-applies this
    # post-retrieval as defense in depth, for documents/backends where the
    # field is absent or the filter can't be pushed down.
    vote_count_clause: dict = {}
    if vote_count_min is not None:
        vote_count_clause["$gte"] = vote_count_min
    if vote_count_max is not None:
        vote_count_clause["$lte"] = vote_count_max
    if vote_count_clause:
        flt["vote_count"] = vote_count_clause
    return flt


async def execute(args: dict) -> dict:
    query_text: str = args["query_text"]
    genres = args.get("genres")
    year_min = args.get("year_min")
    year_max = args.get("year_max")
    min_score = args.get("min_score")
    vote_count_min = args.get("vote_count_min")
    vote_count_max = args.get("vote_count_max")
    niche = bool(args.get("prioritize_niche"))
    top_k = int(args.get("top_k") or config.RAG_TOP_K)
    # Fetch a wider pool than top_k so reranking (below) has real candidates
    # to reorder within, instead of blending within an already-truncated set.
    pool_k = min(max(top_k * _RERANK_POOL_MULTIPLIER, top_k), _RERANK_POOL_MAX)

    try:
        vector = await llm_client.embed(query_text)
    except Exception as exc:
        raise RagUnavailable(f"Embedding call failed: {exc}") from exc

    flt = _build_filter(genres, year_min, year_max, min_score, vote_count_min, vote_count_max)

    use_pinecone = config.RAG_BACKEND == "pinecone"
    try:
        if use_pinecone:
            matches = pinecone_client.query(vector=vector, filter=flt, top_k=pool_k)
        else:
            matches = local_rag_client.search(vector=vector, filter=flt, top_k=pool_k)
    except Exception as exc:
        raise RagUnavailable(f"RAG backend unavailable: {exc}") from exc

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
            "vote_count": md.get("vote_count"),
            "vote_average": md.get("vote_average"),
            "director": md.get("director"),
            "keywords": md.get("keywords", []),
        }
        if use_pinecone:
            result["pinecone_score"] = m.get("score")
        else:
            result["local_score"] = m.get("score")
        results.append(result)

    results = _apply_vote_count_bounds(results, vote_count_min, vote_count_max)
    results = _rerank(results, niche)
    results = results[:top_k]
    return {"results": results, "count": len(results)}
