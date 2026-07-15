"""tmdb_fallback_search — structured discovery via TMDB /discover/movie.

Returns movies that are already filtered by availability (TMDB discover does
this natively when watch_region + watch_providers are supplied), so no separate
availability check is needed for its results.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, List, Optional

from agent.clients.tmdb_client import tmdb_client
from agent.tmdb_mappings import (
    resolve_genre_names,
    resolve_provider_ids_verbose,
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
                        "result must match (AND'd together — a result needs ALL "
                        "of them, not any one), e.g. the user's 'themes' like "
                        "'mixed-race couple' or 'heist'. NEVER put alternative/"
                        "either-or options here (e.g. 'Japan', 'China', 'South "
                        "Korea' for a generic 'East Asia' setting) — AND'ing "
                        "mutually-exclusive terms returns zero results almost "
                        "every time; use keywords_any for alternatives, or "
                        "original_language for a regional-setting constraint. "
                        "Each term is resolved to a TMDB keyword via a fuzzy "
                        "lookup (exact match, substring match, then a retry on "
                        "significant sub-words). If a term still can't be "
                        "resolved, the response reports it in "
                        "'unresolved_keywords_all' and the filter is simply NOT "
                        "applied for that term (it is never silently swapped "
                        "for keywords_any) — check that field and retry with a "
                        "TMDB-vocabulary synonym if it's non-empty. Put the "
                        "defining constraint here, not in keywords_any."
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
                        "'vote_average.desc', 'primary_release_date.desc'. If "
                        "you use 'vote_average.desc' and don't set "
                        "vote_count_min yourself, a floor of 200 is applied "
                        "automatically so a handful of 10/10 votes can't "
                        "outrank genuinely well-regarded films — set your own "
                        "vote_count_min explicitly if you want a different "
                        "floor."
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


def _stopword_subterms(term: str) -> List[str]:
    """Break a multi-word phrase into candidate sub-phrases to retry against
    /search/keyword when the exact phrase has no TMDB keyword (e.g.
    "mixed-race couple" isn't a keyword, but "interracial relationship" is —
    trying individual significant words gives the fuzzy match below a chance
    to find it)."""
    words = [w for w in term.replace("-", " ").split() if w.lower() not in {"a", "an", "the", "of"}]
    subterms: List[str] = []
    if len(words) > 1:
        subterms.append(" ".join(words))
    subterms.extend(words)
    return subterms


async def _lookup_keyword(term: str) -> List[dict]:
    try:
        data = await tmdb_client.search_keyword(term)
    except Exception:
        return []
    return data.get("results") or []


async def _resolve_one_keyword(term: str) -> Optional[int]:
    """Resolve one subject-matter term to a TMDB keyword ID.

    Tries, in order: (1) exact case-insensitive name match on the phrase,
    (2) a substring match either way between the phrase and a keyword name,
    (3) the same two checks against each significant sub-word of the phrase.
    Returns None if nothing plausible is found — the caller must not then
    silently substitute a different (soft) filter.
    """
    wanted = term.strip().lower()
    if not wanted:
        return None

    results = await _lookup_keyword(term)
    for r in results:
        if (r.get("name") or "").strip().lower() == wanted:
            return r["id"]
    for r in results:
        name = (r.get("name") or "").strip().lower()
        if name and (name in wanted or wanted in name):
            return r["id"]

    for sub in _stopword_subterms(term)[1:]:  # skip the already-tried full phrase
        sub_results = await _lookup_keyword(sub)
        sub_wanted = sub.strip().lower()
        for r in sub_results:
            name = (r.get("name") or "").strip().lower()
            if name == sub_wanted or (name and (name in wanted or wanted in name)):
                return r["id"]

    return None


async def _resolve_keyword_ids(terms: List[str]) -> tuple[List[int], List[str]]:
    """Resolve each term to a TMDB keyword ID where possible.

    Returns (resolved_ids, unresolved_terms) — unresolved_terms must be
    surfaced to the caller rather than silently dropped, since dropping a
    hard-constraint term changes what the search actually filters on.
    """
    if not terms:
        return [], []

    resolved = await asyncio.gather(*(_resolve_one_keyword(t) for t in terms))
    ids: List[int] = []
    unresolved: List[str] = []
    for term, kid in zip(terms, resolved):
        if kid is None:
            unresolved.append(term)
        else:
            ids.append(kid)
    return ids, unresolved


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
    # A tiny handful of 10/10 votes can otherwise top a vote_average.desc sort
    # ahead of genuinely well-regarded films — apply a sane floor unless the
    # caller set their own.
    if sort_by == "vote_average.desc" and vote_count_min is None:
        vote_count_min = 200
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
    if platforms and not region:
        # No country was given at all, yet a platform filter was requested.
        # Same underlying problem as unknown_country: without a resolved
        # region, `with_watch_providers` is silently ignored by TMDB and every
        # result would come back unfiltered yet still get stamped
        # "available": True below — that's a lie. Fail loudly instead of
        # returning misleadingly-labeled results.
        return {
            "error": "missing_country",
            "message": (
                f"No country was provided, so availability on {platforms} "
                "cannot be verified. Get the user's country before searching "
                "with a platform filter, or omit `platforms` to search "
                "without availability filtering."
            ),
            "results": [],
            "count": 0,
        }

    provider_ids, unresolved_platforms = resolve_provider_ids_verbose(platforms)
    if platforms and not provider_ids:
        # Without a resolved provider ID, TMDB discover can't filter by
        # platform at all — every result would come back unfiltered yet still
        # get stamped "available": True below. Fail loudly instead (mirrors
        # the unknown_country guard above).
        return {
            "error": "unknown_platform",
            "message": (
                f"Could not resolve platform(s) {unresolved_platforms} to a "
                "TMDB provider ID (see agent/tmdb_mappings.py "
                "PLATFORM_TO_PROVIDER_ID / PLATFORM_NAME_ALIASES). Results "
                "cannot be availability-filtered for them."
            ),
            "results": [],
            "count": 0,
        }

    (keyword_ids_all, unresolved_all), (keyword_ids_any, _unresolved_any) = await asyncio.gather(
        _resolve_keyword_ids(keywords_all), _resolve_keyword_ids(keywords_any)
    )

    filters: dict = {
        "sort_by": sort_by,
        "include_adult": "false",
        # Any access the user can pay for counts as "available" — a rentable
        # or buyable title is still watchable, so don't exclude those tiers.
        "with_watch_monetization_types": "flatrate|free|ads|rent|buy",
    }
    if genres:
        filters["with_genres"] = ",".join(str(g) for g in genres)
    # TMDB's with_keywords doesn't support mixing AND (comma) and OR (pipe) in
    # one query, so a hard constraint (keywords_all) takes priority outright —
    # it must never be diluted by an OR against a merely-nice-to-have term.
    # If a hard term didn't resolve at all, it's simply omitted from the
    # filter (reported via unresolved_all below) rather than silently
    # replaced by the soft keywords_any set.
    if keyword_ids_all:
        filters["with_keywords"] = ",".join(str(k) for k in keyword_ids_all)
    elif keyword_ids_any and not unresolved_all:
        filters["with_keywords"] = "|".join(str(k) for k in keyword_ids_any)
    gte = _year_to_date(year_min, end=False)
    lte = _year_to_date(year_max, end=True)
    if gte:
        filters["primary_release_date.gte"] = gte
    if lte:
        filters["primary_release_date.lte"] = lte
    else:
        # No explicit year_max was given, so default the ceiling to today —
        # a general discovery/"best of" search should never surface an
        # unreleased film (its rating/votes are provisional and meaningless).
        # If the caller explicitly wants a future year_max, that's honored
        # above and this default never applies.
        filters["primary_release_date.lte"] = date.today().isoformat()
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

    out: dict = {"results": results, "count": len(results)}
    if unresolved_all:
        out["unresolved_keywords_all"] = unresolved_all
        out["keyword_filter_applied"] = "none" if not keyword_ids_all else "all"
        out["warning"] = (
            f"Hard keyword(s) {unresolved_all} could not be resolved to a TMDB "
            "keyword — results are NOT filtered by them. Verify the theme "
            "yourself before trusting these results, or retry with a "
            "TMDB-vocabulary synonym."
        )
    elif keyword_ids_all:
        out["keyword_filter_applied"] = "all"
    elif keyword_ids_any:
        out["keyword_filter_applied"] = "any"
    else:
        out["keyword_filter_applied"] = "none"
    return out
