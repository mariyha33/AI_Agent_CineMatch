"""Lookup tables mapping human-readable names to TMDB identifiers.

These are placeholders seeded with a few common entries. Populate the full
lists from TMDB's live endpoints before production use:

  * Providers: GET /watch/providers/movie?watch_region={CC}
  * Regions:   GET /watch/providers/regions
"""
from __future__ import annotations

# Maps a streaming platform display name -> TMDB provider ID.
# TODO: Populate the full list from TMDB /watch/providers/movie endpoint.
PLATFORM_TO_PROVIDER_ID = {
    "Netflix": 8,
    "Amazon Prime Video": 9,
    "Disney+": 337,
    "Apple TV+": 350,
    "HBO Max": 384,
    "Hulu": 15,
    "Paramount+": 531,
}

# Maps a country display name -> ISO 3166-1 alpha-2 region code.
# TODO: Populate the full list.
COUNTRY_TO_REGION_CODE = {
    "Israel": "IL",
    "United States": "US",
    "United Kingdom": "GB",
}


def resolve_provider_ids(platforms: list[str]) -> list[int]:
    """Map platform display names to TMDB provider IDs, skipping unknowns."""
    ids: list[int] = []
    for name in platforms or []:
        pid = PLATFORM_TO_PROVIDER_ID.get(name)
        if pid is not None:
            ids.append(pid)
    return ids


def resolve_region_code(country: str | None) -> str | None:
    """Map a country display name to an ISO region code.

    Accepts a display name (e.g. "Israel") or a raw 2-letter code (e.g. "IL").
    """
    if not country:
        return None
    if country in COUNTRY_TO_REGION_CODE:
        return COUNTRY_TO_REGION_CODE[country]
    if len(country) == 2:  # already looks like an ISO code
        return country.upper()
    return None
