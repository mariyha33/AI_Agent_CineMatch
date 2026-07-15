"""Lookup tables mapping human-readable names to TMDB identifiers.

Seeded with the most common providers/countries/genres. Populate further from
TMDB's live endpoints if needed:

  * Providers: GET /watch/providers/movie?watch_region={CC}
  * Regions:   GET /watch/providers/regions
  * Genres:    GET /genre/movie/list
"""
from __future__ import annotations

# Common alternate spellings/aliases users type for a platform whose TMDB
# display name differs (e.g. the app's own example prompt suggests "Disney+",
# but PLATFORM_TO_PROVIDER_ID's key is "Disney Plus") -> the canonical key in
# PLATFORM_TO_PROVIDER_ID. Lookups in resolve_provider_ids are case-insensitive
# and check this map before giving up.
PLATFORM_NAME_ALIASES = {
    "disney+": "Disney Plus",
    "disney plus": "Disney Plus",
    "apple tv+": "Apple TV Plus",
    "prime video": "Amazon Prime Video",
    "amazon prime": "Amazon Prime Video",
    "hbo max": "HBO Max",
    "max": "HBO Max",
    "paramount+": "Paramount Plus",
    "peacock": "Peacock Premium",
}

# Maps a streaming platform display name -> TMDB provider ID.
PLATFORM_TO_PROVIDER_ID = {
    "Apple TV": 2,
    "Netflix": 8,
    "Amazon Prime Video": 119,  # Last occurrence in dict keys overrides 9
    "MUBI": 11,
    "Hulu": 15,
    "MGM Plus": 34,
    "Rakuten TV": 35,
    "Showtime": 37,
    "BBC iPlayer": 38,
    "HBO": 118,
    "HBO Max": 384,
    "Peacock": 386,
    "Peacock Premium": 387,
}

# Maps a country display name -> ISO 3166-1 alpha-2 region code.
COUNTRY_TO_REGION_CODE = {
    "Andorra": "AD",
    "United Arab Emirates": "AE",
    "Antigua and Barbuda": "AG",
    "Albania": "AL",
    "Argentina": "AR",
    "Austria": "AT",
    "Australia": "AU",
    "Bosnia and Herzegovina": "BA",
    "Barbados": "BB",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Bahrain": "BH",
    "Bermuda": "BM",
    "Bolivia": "BO",
    "Brazil": "BR",
    "Bahamas": "BS",
    "Canada": "CA",
    "Switzerland": "CH",
    "Cote D'Ivoire": "CI",
    "Chile": "CL",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Cuba": "CU",
    "Cape Verde": "CV",
    "Czech Republic": "CZ",
    "Germany": "DE",
    "Denmark": "DK",
    "Dominican Republic": "DO",
    "Algeria": "DZ",
    "Ecuador": "EC",
    "Estonia": "EE",
    "Egypt": "EG",
    "Spain": "ES",
    "Finland": "FI",
    "Fiji": "FJ",
    "France": "FR",
    "United Kingdom": "GB",
    "French Guiana": "GF",
    "Ghana": "GH",
    "Gibraltar": "GI",
    "Guadaloupe": "GP",
    "Equatorial Guinea": "GQ",
    "Greece": "GR",
    "Guatemala": "GT",
    "Hong Kong": "HK",
    "Honduras": "HN",
    "Croatia": "HR",
    "Hungary": "HU",
    "Indonesia": "ID",
    "Ireland": "IE",
    "Israel": "IL",
    "India": "IN",
    "Iraq": "IQ",
    "Iceland": "IS",
    "Italy": "IT",
    "Jamaica": "JM",
    "Jordan": "JO",
    "Japan": "JP",
    "Kenya": "KE",
    "South Korea": "KR",
    "Kuwait": "KW",
    "Lebanon": "LB",
    "St. Lucia": "LC",
    "Liechtenstein": "LI",
    "Lithuania": "LT",
    "Latvia": "LV",
    "Libyan Arab Jamahiriya": "LY",
    "Morocco": "MA",
    "Monaco": "MC",
    "Moldova": "MD",
    "Macedonia": "MK",
    "Malta": "MT",
    "Mauritius": "MU",
    "Mexico": "MX",
    "Malaysia": "MY",
    "Mozambique": "MZ",
    "Niger": "NE",
    "Nigeria": "NG",
    "Netherlands": "NL",
    "Norway": "NO",
    "New Zealand": "NZ",
    "Oman": "OM",
    "Panama": "PA",
    "Peru": "PE",
    "French Polynesia": "PF",
    "Philippines": "PH",
    "Pakistan": "PK",
    "Poland": "PL",
    "Palestinian Territory": "PS",
    "Portugal": "PT",
    "Paraguay": "PY",
    "Qatar": "QA",
    "Romania": "RO",
    "Serbia": "RS",
    "Russia": "RU",
    "Saudi Arabia": "SA",
    "Seychelles": "SC",
    "Sweden": "SE",
    "Singapore": "SG",
    "Slovenia": "SI",
    "Slovakia": "SK",
    "San Marino": "SM",
    "Senegal": "SN",
    "El Salvador": "SV",
    "Turks and Caicos Islands": "TC",
    "Thailand": "TH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Trinidad and Tobago": "TT",
    "Taiwan": "TW",
    "Tanzania": "TZ",
    "Uganda": "UG",
    "United States of America": "US",
    "Uruguay": "UY",
    "Holy See": "VA",
    "Venezuela": "VE",
    "Kosovo": "XK",
    "Yemen": "YE",
    "South Africa": "ZA",
    "Zambia": "ZM"
}

# TMDB's fixed movie genre list (GET /genre/movie/list) — stable IDs, safe to
# hardcode. Used to translate /discover/movie's raw genre_ids into names so
# tmdb_fallback_search results match rag_search's List[str] genre shape.
GENRE_ID_TO_NAME = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}


# Case-insensitive index over PLATFORM_TO_PROVIDER_ID, built once at import.
_PLATFORM_TO_PROVIDER_ID_LOWER = {
    name.lower(): pid for name, pid in PLATFORM_TO_PROVIDER_ID.items()
}


def resolve_provider_ids(platforms: list[str]) -> list[int]:
    """Map platform display names to TMDB provider IDs, skipping unknowns.

    Case-insensitive, and consults PLATFORM_NAME_ALIASES for common alternate
    spellings (e.g. "Disney+") before giving up on a name.
    """
    ids, _unresolved = resolve_provider_ids_verbose(platforms)
    return ids


def resolve_provider_ids_verbose(platforms: list[str]) -> tuple[list[int], list[str]]:
    """Like resolve_provider_ids, but also returns names that couldn't be
    resolved to a TMDB provider ID at all — callers that need to warn (rather
    than silently drop) the filter should use this."""
    ids: list[int] = []
    unresolved: list[str] = []
    for name in platforms or []:
        lowered = name.strip().lower()
        pid = _PLATFORM_TO_PROVIDER_ID_LOWER.get(lowered)
        if pid is None:
            canonical = PLATFORM_NAME_ALIASES.get(lowered)
            if canonical:
                pid = PLATFORM_TO_PROVIDER_ID.get(canonical)
        if pid is not None:
            ids.append(pid)
        else:
            unresolved.append(name)
    return ids, unresolved


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


def resolve_genre_names(genre_ids: list[int]) -> list[str]:
    """Map TMDB genre IDs to names, skipping unknowns."""
    return [GENRE_ID_TO_NAME[g] for g in genre_ids or [] if g in GENRE_ID_TO_NAME]
