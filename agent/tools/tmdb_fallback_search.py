"""tmdb_fallback_search — structured discovery via TMDB /discover/movie.

Returns movies that are already filtered by availability (TMDB discover does
this natively when watch_region + watch_providers are supplied), so no separate
availability check is needed for its results.
"""
from __future__ import annotations

from typing import Any, List, Optional

from agent.clients.tmdb_client import tmdb_client
from agent.tmdb_mappings import resolve_provider_ids, resolve_region_code

TOOL_NAME = "tmdb_fallback_search"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Structured movie discovery via TMDB. Use this to find more obscure "
            "titles, or to honor filters the RAG index can't (e.g. runtime, live "
            "availability). Results are already filtered to be available on the "
            "given platforms in the given country."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "TMDB genre IDs.",
                },
                "year_min": {"type": "integer", "description": "Earliest release year."},
                "year_max": {"type": "integer", "description": "Latest release year."},
                "min_rating": {"type": "number", "description": "Minimum TMDB vote average."},
                "country": {
                    "type": "string",
                    "description": "Country display name or ISO 3166-1 code for availability.",
                },
                "platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Streaming platform display names.",
                },
                "max_results": {"type": "integer", "description": "Cap on results (default 10)."},
            },
            "required": ["country", "platforms"],
        },
    },
}


def _year_to_date(year: Optional[int], end: bool) -> Optional[str]:
    if year is None:
        return None
    return f"{year}-12-31" if end else f"{year}-01-01"


async def execute(args: dict) -> dict:
    genres: List[int] = args.get("genres") or []
    year_min = args.get("year_min")
    year_max = args.get("year_max")
    min_rating = args.get("min_rating")
    country = args.get("country")
    platforms: List[str] = args.get("platforms") or []
    max_results = int(args.get("max_results") or 10)

    region = resolve_region_code(country)
    provider_ids = resolve_provider_ids(platforms)

    filters: dict = {
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "with_watch_monetization_types": "flatrate|free|ads|rent|buy",
    }
    if genres:
        filters["with_genres"] = ",".join(str(g) for g in genres)
    gte = _year_to_date(year_min, end=False)
    lte = _year_to_date(year_max, end=True)
    if gte:
        filters["primary_release_date.gte"] = gte
    if lte:
        filters["primary_release_date.lte"] = lte
    if min_rating is not None:
        filters["vote_average.gte"] = min_rating
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
                "genres": r.get("genre_ids", []),
                "overview": r.get("overview"),
                "rating": r.get("vote_average"),
                "available": True,
                "platform": platform_label,
                "country": region,
            }
        )
    return {"results": results, "count": len(results)}
