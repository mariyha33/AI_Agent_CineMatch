"""verify_recommendation — ground the Reflection critique in live TMDB data.

For each candidate (in parallel): fetch movie detail + current watch providers.
The tool itself decides availability (verdict pass/fail); the LLM uses the
remaining data (genres, overview, popularity, keywords) to judge taste/novelty.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from agent.clients.tmdb_client import tmdb_client
from agent.tmdb_mappings import (
    PLATFORM_TO_PROVIDER_ID,
    resolve_provider_ids,
    resolve_region_code,
)

TOOL_NAME = "verify_recommendation"


class TMDBUnavailable(RuntimeError):
    """Raised when TMDB appears entirely unreachable (every candidate failed)."""


SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Verify a draft list of candidates against live TMDB data. Returns, "
            "per candidate, whether it is available on the requested platforms in "
            "the requested country (with a pass/fail verdict), plus genres, "
            "overview, popularity, and keyword tags for taste assessment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tmdb_id": {"type": "integer"},
                            "title": {"type": "string"},
                        },
                        "required": ["tmdb_id"],
                    },
                    "description": "The candidates to verify.",
                },
                "country": {"type": "string", "description": "Country display name or ISO code."},
                "platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Streaming platform display names to check.",
                },
                "user_mood": {
                    "type": "string",
                    "description": "The original mood/taste description, for comparison.",
                },
                "exclude_people": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directors/actors to check the credits for and fail the candidate if present.",
                },
            },
            "required": ["candidates", "country", "platforms"],
        },
    },
}


def _collect_region_providers(providers_payload: dict, region: Optional[str]) -> List[dict]:
    """Flatten all provider entries for a region across monetization types."""
    results = (providers_payload or {}).get("results", {}) or {}
    region_data = results.get(region or "", {}) or {}
    entries: List[dict] = []
    for key in ("flatrate", "free", "ads", "rent", "buy"):
        entries.extend(region_data.get(key, []) or [])
    return entries


def _match_platform(
    region_providers: List[dict],
    wanted_platform_names: List[str],
    wanted_provider_ids: List[int],
) -> Optional[str]:
    """Return the matching platform display name, or None if unavailable."""
    wanted_ids = set(wanted_provider_ids)
    wanted_names = {n.lower() for n in wanted_platform_names}
    for p in region_providers:
        pid = p.get("provider_id")
        pname = (p.get("provider_name") or "").lower()
        if pid in wanted_ids or pname in wanted_names:
            return p.get("provider_name")
    return None


def _excluded_person_in_credits(
    credits: dict, excluded_lower: set, cast_limit: int = 5
) -> Optional[str]:
    """Return the name of an excluded director/top-billed cast member found in
    the credits, or None. Only checks the director and the first `cast_limit`
    billed cast members — a person appearing deep in the credits isn't what a
    user means by "starring"."""
    for c in credits.get("crew") or []:
        if (c.get("job") or "").strip().lower() == "director":
            name = (c.get("name") or "").strip()
            if name.lower() in excluded_lower:
                return name
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))
    for c in cast[:cast_limit]:
        name = (c.get("name") or "").strip()
        if name.lower() in excluded_lower:
            return name
    return None


async def _verify_one(
    candidate: dict,
    region: Optional[str],
    platforms: List[str],
    provider_ids: List[int],
    excluded_people_lower: set,
) -> Optional[dict]:
    tmdb_id = candidate.get("tmdb_id")
    title = candidate.get("title")
    try:
        if excluded_people_lower:
            movie, providers, keywords, credits = await asyncio.gather(
                tmdb_client.get_movie(tmdb_id),
                tmdb_client.get_watch_providers(tmdb_id),
                tmdb_client.get_movie_keywords(tmdb_id),
                tmdb_client.get_movie_credits(tmdb_id),
            )
        else:
            movie, providers, keywords = await asyncio.gather(
                tmdb_client.get_movie(tmdb_id),
                tmdb_client.get_watch_providers(tmdb_id),
                tmdb_client.get_movie_keywords(tmdb_id),
            )
            credits = None
    except Exception:
        # Skip this candidate on any TMDB error (§12).
        return None

    region_providers = _collect_region_providers(providers, region)
    matched_platform = _match_platform(region_providers, platforms, provider_ids)
    available = matched_platform is not None

    genres = [g.get("name") for g in movie.get("genres", []) if g.get("name")]
    keyword_tags = [k.get("name") for k in (keywords.get("keywords") or []) if k.get("name")]

    excluded_person = (
        _excluded_person_in_credits(credits, excluded_people_lower)
        if credits is not None
        else None
    )

    if excluded_person:
        available = False
        matched_platform = None
        verdict, reason = "fail", f"Involves excluded person: {excluded_person}."
    elif available:
        verdict, reason = "pass", f"Available on {matched_platform} in {region}."
    else:
        verdict, reason = "fail", (
            f"Not available on the requested platforms ({', '.join(platforms) or 'none'}) "
            f"in {region or 'the requested country'}."
        )

    return {
        "tmdb_id": tmdb_id,
        "title": title or movie.get("title"),
        "available": available,
        "platform": matched_platform,
        "genres": genres,
        "overview": movie.get("overview"),
        "popularity": movie.get("popularity"),
        # A far better mainstream-ness proxy than `popularity` (a unitless,
        # constantly-rescaled TMDB metric) — see REFLECTION_AGENT_SYSTEM_PROMPT.
        "vote_count": movie.get("vote_count"),
        "vote_average": movie.get("vote_average"),
        "keyword_tags": keyword_tags[:15],
        "verdict": verdict,
        "reason": reason,
    }


async def execute(args: dict) -> dict:
    candidates: List[dict] = args.get("candidates") or []
    country = args.get("country")
    platforms: List[str] = args.get("platforms") or []
    exclude_people: List[str] = args.get("exclude_people") or []
    excluded_people_lower = {p.strip().lower() for p in exclude_people if p and p.strip()}

    region = resolve_region_code(country)
    provider_ids = resolve_provider_ids(platforms)

    if platforms and not region:
        # Without a resolved region, availability can't be checked at all —
        # every candidate would otherwise come back "fail" with a misleading
        # per-movie reason ("not available in the requested country") when
        # the real problem is that no usable country was ever given. Fail
        # loudly at the batch level instead of individually and silently
        # (mirrors tmdb_fallback_search.py's equivalent guard).
        return {
            "error": "missing_country" if not country else "unknown_country",
            "message": (
                f"Cannot verify availability on {platforms} without a "
                f"resolvable country (got country={country!r})."
            ),
            "results": [],
            "count": 0,
            "region": region,
        }

    tasks = [
        _verify_one(c, region, platforms, provider_ids, excluded_people_lower)
        for c in candidates
    ]
    settled = await asyncio.gather(*tasks)

    results = [r for r in settled if r is not None]

    # If we had candidates but every single one errored out (a genuine TMDB
    # request failure — see _verify_one's except clause), TMDB is likely down.
    # This must NOT fire just because candidates came back unavailable or
    # below the caller's match bar: _verify_one still returns a normal dict
    # with verdict="fail" for those (they land in `results`), it only returns
    # None on a real request exception. So `results` being empty here means
    # every request itself failed, not "nothing passed availability" — the
    # caller (agent/orchestrator.py) additionally makes sure this exception
    # never discards candidates a PREVIOUS batch already verified/approved.
    if candidates and not results:
        raise TMDBUnavailable(
            "Movie database (TMDB) is currently unavailable. Please try again later."
        )

    return {"results": results, "count": len(results), "region": region}
