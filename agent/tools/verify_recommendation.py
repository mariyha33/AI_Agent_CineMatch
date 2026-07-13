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


async def _verify_one(
    candidate: dict,
    region: Optional[str],
    platforms: List[str],
    provider_ids: List[int],
) -> Optional[dict]:
    tmdb_id = candidate.get("tmdb_id")
    title = candidate.get("title")
    try:
        movie, providers, keywords = await asyncio.gather(
            tmdb_client.get_movie(tmdb_id),
            tmdb_client.get_watch_providers(tmdb_id),
            tmdb_client.get_movie_keywords(tmdb_id),
        )
    except Exception:
        # Skip this candidate on any TMDB error (§12).
        return None

    region_providers = _collect_region_providers(providers, region)
    matched_platform = _match_platform(region_providers, platforms, provider_ids)
    available = matched_platform is not None

    genres = [g.get("name") for g in movie.get("genres", []) if g.get("name")]
    keyword_tags = [k.get("name") for k in (keywords.get("keywords") or []) if k.get("name")]

    if available:
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
        "keyword_tags": keyword_tags[:15],
        "verdict": verdict,
        "reason": reason,
    }


async def execute(args: dict) -> dict:
    candidates: List[dict] = args.get("candidates") or []
    country = args.get("country")
    platforms: List[str] = args.get("platforms") or []

    region = resolve_region_code(country)
    provider_ids = resolve_provider_ids(platforms)

    tasks = [_verify_one(c, region, platforms, provider_ids) for c in candidates]
    settled = await asyncio.gather(*tasks)

    results = [r for r in settled if r is not None]

    # If we had candidates but every single one errored out, TMDB is likely down.
    if candidates and not results:
        raise TMDBUnavailable(
            "Movie database (TMDB) is currently unavailable. Please try again later."
        )

    return {"results": results, "count": len(results), "region": region}
