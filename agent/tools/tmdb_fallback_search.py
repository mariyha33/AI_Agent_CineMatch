"""tmdb_fallback_search — structured discovery via TMDB /discover/movie.

Returns movies that are already filtered by availability (TMDB discover does
this natively when watch_region + watch_providers are supplied), so no separate
availability check is needed for its results.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from agent.clients.tmdb_client import tmdb_client
from agent.tmdb_mappings import (
    resolve_genre_names,
    resolve_provider_ids,
    resolve_region_code,
)

TOOL_NAME = "tmdb_fallback_search"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Structured movie discovery via TMDB. Use this to find more obscure "
            "titles, or to honor filters the RAG index can't (e.g. runtime, live "
            "availability). Results are already filtered to be available on the "
            "given platforms in the given country.\n"
            "For a NICHE / 'not popular' request, set vote_count_max (e.g. 1000) "
            "to exclude blockbusters, keep a small vote_count_min (e.g. 20) to "
            "avoid completely obscure/unrated titles, and consider sort_by="
            "'vote_average.desc' so quality — not popularity — ranks results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "TMDB genre IDs.",
                },
                "keywords_all": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Non-negotiable subject-matter/plot terms that EVERY "
                        "result must match (AND'd together), e.g. the user's "
                        "'themes' like 'mixed-race couple' or 'heist'. Each is "
                        "resolved to a TMDB keyword; unmatched terms are "
                        "ignored. Put the defining constraint here, not in "
                        "keywords_any — if this returns 0 results, retry with "
                        "fewer/broader terms rather than moving them to "
                        "keywords_any."
                    ),
                },
                "keywords_any": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Nice-to-have mood/tone terms (e.g. 'quirky', 'slow "
                        "burn') — a result needs to match only ONE of these "
                        "(OR'd together). Use for flavor, not for the request's "
                        "defining subject matter."
                    ),
                },
                "year_min": {"type": "integer", "description": "Earliest release year."},
                "year_max": {"type": "integer", "description": "Latest release year."},
                "min_rating": {"type": "number", "description": "Minimum TMDB vote average."},
                "vote_count_min": {
                    "type": "integer",
                    "description": "Minimum number of TMDB votes (filters out unrated/obscure noise).",
                },
                "vote_count_max": {
                    "type": "integer",
                    "description": "Maximum number of TMDB votes. Use a low value (e.g. 500-1500) for niche/not-popular requests.",
                },
                "runtime_min": {"type": "integer", "description": "Minimum runtime in minutes."},
                "runtime_max": {"type": "integer", "description": "Maximum runtime in minutes."},
                "original_language": {
                    "type": "string",
                    "description": "ISO 639-1 language code (e.g. 'en', 'fr') to restrict original language.",
                },
                "sort_by": {
                    "type": "string",
                    "description": (
                        "TMDB sort order, e.g. 'popularity.desc' (default), "
                        "'vote_average.desc', 'primary_release_date.desc'."
                    ),
                },
                "country": {
                    "type": "string",
                    "description": (
                        "Country display name or ISO 3166-1 code for availability. "
                        "Auto-filled from the user's preferences — you don't need "
                        "to supply this."
                    ),
                },
                "platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Streaming platform display names. Auto-filled from the "
                        "user's preferences — you don't need to supply this."
                    ),
                },
                "max_results": {"type": "integer", "description": "Cap on results (default 10)."},
            },
            "required": [],
        },
    },
}


def _year_to_date(year: Optional[int], end: bool) -> Optional[str]:
    if year is None:
        return None
    return f"{year}-12-31" if end else f"{year}-01-01"


async def _resolve_keyword_ids(terms: List[str]) -> List[int]:
    if not terms:
        return []

    async def _lookup(term: str) -> Optional[int]:
        try:
            data = await tmdb_client.search_keyword(term)
        except Exception:
            return None
        results = data.get("results") or []
        if not results:
            return None
        # Prefer an exact (case-insensitive) name match over TMDB's top hit —
        # search_keyword is a free-text search, so the first result isn't
        # necessarily the term the caller meant.
        wanted = term.strip().lower()
        for r in results:
            if (r.get("name") or "").strip().lower() == wanted:
                return r["id"]
        return results[0]["id"]

    resolved = await asyncio.gather(*(_lookup(t) for t in terms))
    return [kid for kid in resolved if kid is not None]


async def execute(args: dict) -> dict:
    genres: List[int] = args.get("genres") or []
    keywords_all: List[str] = args.get("keywords_all") or []
    keywords_any: List[str] = args.get("keywords_any") or []
    year_min = args.get("year_min")
    year_max = args.get("year_max")
    min_rating = args.get("min_rating")
    vote_count_min = args.get("vote_count_min")
    vote_count_max = args.get("vote_count_max")
    runtime_min = args.get("runtime_min")
    runtime_max = args.get("runtime_max")
    original_language = args.get("original_language")
    sort_by = args.get("sort_by") or "popularity.desc"
    country = args.get("country")
    platforms: List[str] = args.get("platforms") or []
    max_results = int(args.get("max_results") or 10)

    region = resolve_region_code(country)
    if country and not region:
        # Without a resolved region, TMDB discover can't filter by
        # availability at all — every result would come back unfiltered yet
        # still get stamped "available": True below. Fail loudly instead.
        return {
            "error": "unknown_country",
            "message": (
                f"Could not resolve country '{country}' to a TMDB region code "
                "(see agent/tmdb_mappings.py COUNTRY_TO_REGION_CODE). Results "
                "cannot be availability-filtered for it."
            ),
            "results": [],
            "count": 0,
        }

    provider_ids = resolve_provider_ids(platforms)
    keyword_ids_all, keyword_ids_any = await asyncio.gather(
        _resolve_keyword_ids(keywords_all), _resolve_keyword_ids(keywords_any)
    )

    filters: dict = {
        "sort_by": sort_by,
        "include_adult": "false",
        # "rent"/"buy" are pay-per-title, not "available on <platform>" in the
        # subscription sense the user means — only count included tiers.
        "with_watch_monetization_types": "flatrate|free|ads",
    }
    if genres:
        filters["with_genres"] = ",".join(str(g) for g in genres)
    # TMDB's with_keywords doesn't support mixing AND (comma) and OR (pipe) in
    # one query, so a hard constraint (keywords_all) takes priority outright —
    # it must never be diluted by an OR against a merely-nice-to-have term.
    if keyword_ids_all:
        filters["with_keywords"] = ",".join(str(k) for k in keyword_ids_all)
    elif keyword_ids_any:
        filters["with_keywords"] = "|".join(str(k) for k in keyword_ids_any)
    gte = _year_to_date(year_min, end=False)
    lte = _year_to_date(year_max, end=True)
    if gte:
        filters["primary_release_date.gte"] = gte
    if lte:
        filters["primary_release_date.lte"] = lte
    if min_rating is not None:
        filters["vote_average.gte"] = min_rating
    if vote_count_min is not None:
        filters["vote_count.gte"] = vote_count_min
    if vote_count_max is not None:
        filters["vote_count.lte"] = vote_count_max
    if runtime_min is not None:
        filters["with_runtime.gte"] = runtime_min
    if runtime_max is not None:
        filters["with_runtime.lte"] = runtime_max
    if original_language:
        filters["with_original_language"] = original_language
    if region:
        filters["watch_region"] = region
    if provider_ids:
        filters["with_watch_providers"] = "|".join(str(p) for p in provider_ids)

    data = await tmdb_client.discover_movies(filters)
    raw = (data.get("results") or [])[:max_results]

    # Report the platform label back (first requested platform we recognized).
    platform_label = platforms[0] if platforms else None

    results: List[dict] = []
    for r in raw:
        release = r.get("release_date") or ""
        year = int(release[:4]) if len(release) >= 4 and release[:4].isdigit() else None
        results.append(
            {
                "title": r.get("title"),
                "year": year,
                "tmdb_id": r.get("id"),
                "genres": resolve_genre_names(r.get("genre_ids", [])),
                "overview": r.get("overview"),
                "rating": r.get("vote_average"),
                "available": True,
                "platform": platform_label,
                "country": region,
            }
        )
    return {"results": results, "count": len(results)}
