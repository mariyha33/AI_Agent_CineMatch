#!/usr/bin/env python3
"""
build_canonical_movies.py

Phase 1 of the CineMatch data pipeline: build a clean, TMDB-verified
"canonical" movie table from the raw IMDB CSV export.

This script does NOT build the RAG, does NOT create embeddings, does NOT
build the AI agent, and does NOT call any LLM. It only cleans the raw data
and resolves each movie to a verified TMDB id, because:

    No tmdb_id -> no embedding -> no recommendation.

Every row in data/imdb_movies.csv ends up in exactly one of the two output
tables: canonical_movies.csv (verified, high-confidence TMDB match) or
unmatched_movies.csv (dropped during cleaning, or no confident TMDB match).
Nothing is silently discarded.

Usage:
    # Option A: environment variable
    export TMDB_API_KEY=your_v3_api_key   # https://www.themoviedb.org/settings/api
    python3 scripts/build_canonical_movies.py

    # Option B: .env file (requires `pip install python-dotenv`, loaded automatically)
    cp .env.example .env && edit .env to set TMDB_API_KEY
    python3 scripts/build_canonical_movies.py

    python3 scripts/build_canonical_movies.py --limit 200   # smoke test
    python3 scripts/build_canonical_movies.py --review-unmatched
    python3 scripts/build_canonical_movies.py --no-cache --workers 8

See README.md ("Phase 1 — Canonical Movie Table") for full documentation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Load a .env file if present, so `cp .env.example .env` + editing it is
# enough to configure the script - no manual `export` required. This is a
# soft dependency: if python-dotenv isn't installed, we just skip it and
# fall back to whatever's already in the real environment / a manual export.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Fuzzy string similarity: prefer rapidfuzz (faster, better normalization)
# but fall back to stdlib difflib so this script has zero hard dependencies
# beyond pandas + requests.
# --------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz

    def similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return fuzz.token_sort_ratio(a, b) / 100.0

except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# --------------------------------------------------------------------------
# Configuration (env vars per spec section 5 - never hardcode secrets)
# --------------------------------------------------------------------------
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE_URL = os.environ.get("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_API_READ_TOKEN = os.environ.get("TMDB_API_READ_TOKEN", "")  # optional v4 bearer alt-auth

DEFAULT_INPUT_CSV = os.environ.get("CANONICAL_INPUT_CSV", "data/imdb_movies.csv")
DEFAULT_OUTPUT_CSV = os.environ.get("CANONICAL_OUTPUT_CSV", "data/processed/canonical_movies.csv")
DEFAULT_UNMATCHED_CSV = os.environ.get("CANONICAL_UNMATCHED_CSV", "data/processed/unmatched_movies.csv")
DEFAULT_REPORT_JSON = os.environ.get("CANONICAL_REPORT_JSON", "data/processed/canonical_build_report.json")
DEFAULT_CACHE_PATH = os.environ.get("TMDB_CACHE_PATH", "data/cache/tmdb_search_cache.json")
CONFIDENCE_THRESHOLD = float(os.environ.get("TMDB_MATCH_CONFIDENCE_THRESHOLD", "0.82"))
REQUEST_SLEEP_SECONDS = float(os.environ.get("TMDB_REQUEST_SLEEP_SECONDS", "0.25"))

# Internal, non-configurable floors/margins (documented here, not in .env,
# because they're implementation details of the scoring model rather than
# per-environment tuning knobs).
MIN_TITLE_SIM = 0.50        # below this, a candidate is not "plausible" at all
AMBIGUOUS_MARGIN = 0.04     # #1 vs #2 candidate score gap below this = ambiguous
MAX_YEAR_DIFF = 1
MIN_OVERVIEW_LEN = 10       # shorter than this counts as "missing overview"

# Confidence scoring weights (spec section 8) - must sum to 1.0
W_TITLE = 0.45
W_YEAR = 0.25
W_LANG = 0.15
W_OVERVIEW = 0.15

# IMDB CSV language *names* -> ISO 639-1 codes used by TMDB's original_language.
# Used only as a soft signal (bonus/penalty), never to disqualify a match on
# its own - some entries (e.g. "No Language") intentionally map to None.
LANG_NAME_TO_CODE = {
    "arabic": "ar", "basque": "eu", "bengali": "bn",
    "bokmål, norwegian, norwegian bokmål": "nb", "cantonese": "cn",
    "catalan, valencian": "ca", "central khmer": "km", "chinese": "zh",
    "czech": "cs", "danish": "da", "dutch, flemish": "nl", "dzongkha": "dz",
    "english": "en", "finnish": "fi", "french": "fr", "galician": "gl",
    "german": "de", "greek": "el", "gujarati": "gu", "hindi": "hi",
    "hungarian": "hu", "icelandic": "is", "indonesian": "id", "irish": "ga",
    "italian": "it", "japanese": "ja", "kannada": "kn", "korean": "ko",
    "latin": "la", "latvian": "lv", "macedonian": "mk", "malay": "ms",
    "malayalam": "ml", "marathi": "mr", "no language": None,
    "norwegian": "no", "oriya": "or", "persian": "fa", "polish": "pl",
    "portuguese": "pt", "romanian": "ro", "russian": "ru", "serbian": "sr",
    "serbo-croatian": "sh", "slovak": "sk", "spanish, castilian": "es",
    "swedish": "sv", "tagalog": "tl", "tamil": "ta", "telugu": "te",
    "thai": "th", "turkish": "tr", "ukrainian": "uk", "vietnamese": "vi",
}


# --------------------------------------------------------------------------
# Normalization helpers (spec section 6)
# --------------------------------------------------------------------------
def normalize_text(value) -> str:
    """Strip whitespace and fix mojibake artifacts like the U+00A0 non-breaking
    space (which shows up as a stray 'Â' when the CSV is misread as Latin-1)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value)
    s = s.replace("\xa0", " ")   # the actual broken character in this dataset
    s = s.replace("Â", "")       # defensive: literal mojibake if it ever appears
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_title(title: str) -> str:
    """Lowercased, punctuation-stripped title key, used for de-duplication."""
    t = normalize_text(title).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t


def normalize_genres(value) -> str:
    raw = normalize_text(value)
    if not raw:
        return ""
    parts = [normalize_text(p) for p in raw.split(",")]
    parts = [p for p in parts if p]
    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return ", ".join(out)


def normalize_language(orig_lang: str) -> Optional[str]:
    """Map an IMDB CSV language *name* (e.g. 'English') to an ISO 639-1 code
    (e.g. 'en') comparable to TMDB's original_language. Returns None if the
    name isn't recognized (treated as neutral/unknown, never disqualifying)."""
    key = normalize_text(orig_lang).lower()
    return LANG_NAME_TO_CODE.get(key)


def extract_release_year(date_x) -> Optional[int]:
    s = normalize_text(date_x)
    if not s:
        return None
    dt = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce")  # last-resort generic parse
    if pd.isna(dt):
        return None
    return int(dt.year)


def dedupe_key(title: str, year: Optional[int]) -> str:
    return f"{normalize_title(title)}|{year if year is not None else ''}"


def candidate_year(candidate: dict) -> Optional[int]:
    rd = candidate.get("release_date") or ""
    if len(rd) >= 4:
        try:
            return int(rd[:4])
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------
# TMDB client with local disk cache + retry/backoff
# --------------------------------------------------------------------------
class TmdbClient:
    def __init__(self, cache_path: Path, use_cache: bool = True, min_interval: float = REQUEST_SLEEP_SECONDS):
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.min_interval = min_interval
        self._lock = Lock()
        self._last_call = 0.0
        self._cache: dict = {}
        if use_cache and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self._session = requests.Session()

    def _auth_params_and_headers(self):
        if TMDB_API_READ_TOKEN:
            return {}, {"Authorization": f"Bearer {TMDB_API_READ_TOKEN}"}
        if TMDB_API_KEY:
            return {"api_key": TMDB_API_KEY}, {}
        raise RuntimeError(
            "No TMDB credentials found. Set TMDB_API_KEY (v3 key) or "
            "TMDB_API_READ_TOKEN (v4 bearer token) as an environment variable."
        )

    def _throttle(self):
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()

    def _get(self, path: str, params: dict) -> dict:
        auth_params, headers = self._auth_params_and_headers()
        request_params = {**params, **auth_params}  # only this merged copy ever holds api_key
        max_retries = 5
        backoff = 1.0
        for _ in range(max_retries):
            self._throttle()
            try:
                resp = self._session.get(f"{TMDB_BASE_URL}{path}", params=request_params, headers=headers, timeout=15)
            except requests.RequestException:
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                time.sleep(retry_after)
                backoff *= 2
                continue
            if resp.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        # Log only the original, safe params (query/year/etc.) - never the
        # auth-merged request_params, which would include api_key.
        raise RuntimeError(f"TMDB request failed after {max_retries} retries: {path} {params}")

    def search_movie(self, query: str, year: Optional[int] = None) -> list:
        cache_key = f"search::{query}::{year or ''}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        params = {"query": query, "include_adult": "false", "language": "en-US"}
        if year:
            # Search year-scoped first (cheaper in the common case): if TMDB
            # finds something for this exact primary_release_year, we're done
            # in one call. Only fall back to an unscoped search - a second
            # API call - when the year-scoped search comes back empty (e.g.
            # our release_year is off, or TMDB's primary_release_year filter
            # missed a valid match).
            results = self._get("/search/movie", {**params, "primary_release_year": year}).get("results", [])
            if not results:
                results = self._get("/search/movie", params).get("results", [])
        else:
            results = self._get("/search/movie", params).get("results", [])

        if self.use_cache:
            self._cache[cache_key] = results
        return results

    def movie_details(self, tmdb_id: int) -> dict:
        cache_key = f"details::{tmdb_id}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        details = self._get(f"/movie/{tmdb_id}", {"language": "en-US"})
        if self.use_cache:
            self._cache[cache_key] = details
        return details

    def save_cache(self):
        if not self.use_cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")


# --------------------------------------------------------------------------
# Matching (spec sections 7-8)
# --------------------------------------------------------------------------
@dataclass
class MatchResult:
    matched: bool
    tmdb_id: Optional[int] = None
    confidence: float = 0.0
    reason: str = ""
    candidate: Optional[dict] = None
    best_candidate_title: Optional[str] = None
    best_candidate_original_title: Optional[str] = None
    best_candidate_year: Optional[int] = None
    best_candidate_tmdb_id: Optional[int] = None
    best_candidate_confidence: float = 0.0


def score_candidate(candidate: dict, names: str, orig_title: str, year: Optional[int],
                     lang_code: Optional[str], source_overview: str):
    """Returns (total_score, components) or (-1.0, {}) if hard-disqualified
    by the year cutoff. `total_score` is the weighted 0-1 confidence."""
    cand_title = candidate.get("title") or ""
    cand_orig_title = candidate.get("original_title") or ""
    title_sim = max(
        similarity(names, cand_title),
        similarity(names, cand_orig_title),
        similarity(orig_title, cand_title),
        similarity(orig_title, cand_orig_title),
    )

    cand_yr = candidate_year(candidate)
    if year is not None and cand_yr is not None:
        diff = abs(cand_yr - year)
        if diff > MAX_YEAR_DIFF:
            return -1.0, {}
        year_score = 1.0 if diff == 0 else 0.6
        year_exact = diff == 0
    else:
        year_score = 0.5  # neutral: don't disqualify on missing data alone
        year_exact = None

    cand_lang = candidate.get("original_language")
    if lang_code and cand_lang:
        lang_score = 1.0 if lang_code == cand_lang else 0.0
        lang_compatible = lang_code == cand_lang
    else:
        lang_score = 0.5  # unknown: neutral
        lang_compatible = None

    cand_overview = candidate.get("overview") or ""
    overview_sim = similarity(source_overview, cand_overview) if source_overview and cand_overview else 0.5

    total = W_TITLE * title_sim + W_YEAR * year_score + W_LANG * lang_score + W_OVERVIEW * overview_sim
    components = {
        "title_sim": title_sim,
        "year_score": year_score,
        "year_exact": year_exact,
        "lang_score": lang_score,
        "lang_compatible": lang_compatible,
        "overview_sim": overview_sim,
    }
    return total, components


def describe_accept_reason(source_label: str, components: dict) -> str:
    title_part = f"{'original title' if source_label == 'orig_title' else 'title'} similarity {components['title_sim']:.2f}"
    if components["year_exact"] is True:
        year_part = "exact release year match"
    elif components["year_exact"] is False:
        year_part = "release year differs by 1"
    else:
        year_part = "release year unknown"
    if components["lang_compatible"] is True:
        lang_part = "language compatible"
    elif components["lang_compatible"] is False:
        lang_part = "language mismatch"
    else:
        lang_part = "language unknown"
    return f"Accepted: {title_part}, {year_part}, {lang_part}."


def match_movie(client: TmdbClient, names: str, orig_title: str, year: int,
                 lang_code: Optional[str], source_overview: str) -> MatchResult:
    attempts = [("names", names), ("orig_title", orig_title)]
    overall_best = None  # (score, candidate, source_label, components)
    any_raw_candidates = False
    any_year_survivors = False

    for source_label, query in attempts:
        query = normalize_text(query)
        if not query:
            continue
        try:
            candidates = client.search_movie(query, year)
        except RuntimeError:
            raise
        except Exception:
            candidates = []

        if candidates:
            any_raw_candidates = True

        scored = []
        for c in candidates:
            s, comps = score_candidate(c, names, orig_title, year, lang_code, source_overview)
            if s >= 0:
                scored.append((s, c, comps))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            continue
        any_year_survivors = True

        top_score, top_candidate, top_comps = scored[0]
        if overall_best is None or top_score > overall_best[0]:
            overall_best = (top_score, top_candidate, source_label, top_comps)

        if top_comps["title_sim"] < MIN_TITLE_SIM:
            continue  # not plausible, try next attempt

        if len(scored) > 1:
            second_score, second_candidate, _ = scored[1]
            if (top_score - second_score) < AMBIGUOUS_MARGIN and second_candidate.get("id") != top_candidate.get("id"):
                if top_score >= CONFIDENCE_THRESHOLD:
                    return MatchResult(
                        matched=False,
                        confidence=round(top_score, 4),
                        reason="Ambiguous candidates with similar confidence.",
                        best_candidate_title=top_candidate.get("title"),
                        best_candidate_original_title=top_candidate.get("original_title"),
                        best_candidate_year=candidate_year(top_candidate),
                        best_candidate_tmdb_id=top_candidate.get("id"),
                        best_candidate_confidence=round(top_score, 4),
                    )
                continue  # low-confidence AND ambiguous, try next attempt

        if top_score < CONFIDENCE_THRESHOLD:
            continue  # plausible but not confident enough, try next attempt

        return MatchResult(
            matched=True,
            tmdb_id=top_candidate.get("id"),
            confidence=round(top_score, 4),
            reason=describe_accept_reason(source_label, top_comps),
            candidate=top_candidate,
        )

    if overall_best is not None:
        score, cand, source_label, comps = overall_best
        if comps["title_sim"] < MIN_TITLE_SIM:
            reason = "Best title similarity below threshold."
        else:
            reason = f"Confidence below threshold ({score:.2f} < {CONFIDENCE_THRESHOLD:.2f})."
        return MatchResult(
            matched=False,
            confidence=round(max(score, 0.0), 4),
            reason=reason,
            best_candidate_title=cand.get("title"),
            best_candidate_original_title=cand.get("original_title"),
            best_candidate_year=candidate_year(cand),
            best_candidate_tmdb_id=cand.get("id"),
            best_candidate_confidence=round(max(score, 0.0), 4),
        )

    if any_raw_candidates and not any_year_survivors:
        return MatchResult(matched=False, confidence=0.0, reason="Best candidate year differs by more than 1 year.")

    return MatchResult(matched=False, confidence=0.0, reason="No TMDB candidates found.")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def deduplicate_by_tmdb_id(canonical_rows: list) -> tuple:
    """Post-matching safety net: two different source rows can independently
    resolve to the same tmdb_id (e.g. a near-duplicate title in the raw CSV
    that survived cleaning as a distinct row, or an ambiguous alternate
    title). canonical_movies.csv must contain each tmdb_id exactly once, so
    for any tmdb_id with more than one row we keep the single best one and
    demote the rest into unmatched_movies.csv with reason
    'duplicate_tmdb_id', rather than silently dropping or failing validation.

    Tie-break order for "best" when multiple rows share a tmdb_id:
      1. highest match_confidence
      2. highest title/original_title similarity against the canonical name
      3. lowest source_row_id (deterministic - first row wins)
    """
    groups: dict = {}
    for row in canonical_rows:
        groups.setdefault(row["tmdb_id"], []).append(row)

    def title_match_score(row: dict) -> float:
        return max(
            similarity(row.get("source_name", ""), row.get("title", "")),
            similarity(row.get("source_original_title", ""), row.get("original_title", "")),
        )

    kept = []
    demoted = []
    for tmdb_id, rows in groups.items():
        if len(rows) == 1:
            kept.append(rows[0])
            continue

        rows_sorted = sorted(
            rows,
            key=lambda r: (-r["match_confidence"], -title_match_score(r), r["source_row_id"]),
        )
        best = rows_sorted[0]
        kept.append(best)
        for loser in rows_sorted[1:]:
            demoted.append({
                "source_row_id": loser["source_row_id"],
                "names": loser["source_name"],
                "orig_title": loser["source_original_title"],
                "release_year": loser["release_year"],
                "reason": "duplicate_tmdb_id",
                "unmatched_reason": "duplicate_tmdb_id",
                "best_candidate_title": loser["title"],
                "best_candidate_original_title": loser["original_title"],
                "best_candidate_year": loser["release_year"],
                "best_candidate_tmdb_id": loser["tmdb_id"],
                "best_candidate_confidence": loser["match_confidence"],
                "duplicate_of_tmdb_id": tmdb_id,
            })

    kept.sort(key=lambda r: r["source_row_id"])
    demoted.sort(key=lambda r: r["source_row_id"])
    return kept, demoted


CANONICAL_COLUMNS = [
    "source_row_id", "tmdb_id", "title", "original_title", "release_date",
    "release_year", "genres", "overview", "score", "original_language",
    "runtime", "source_name", "source_original_title", "source_genre",
    "source_overview", "source_orig_lang", "source_country",
    "match_confidence", "match_reason",
]

UNMATCHED_COLUMNS = [
    "source_row_id", "names", "orig_title", "release_year", "reason",
    "unmatched_reason",
    "best_candidate_title", "best_candidate_original_title",
    "best_candidate_year", "best_candidate_tmdb_id", "best_candidate_confidence",
    "duplicate_of_tmdb_id",
]


def load_and_triage(input_path: Path):
    """Cleans the raw CSV and splits every row into either 'survivors' (go on
    to TMDB matching) or immediate 'unmatched' entries (dropped during
    cleaning, with a reason) - per spec section 11, cleaning-time drops must
    also show up in unmatched_movies.csv, not be silently discarded."""
    df = pd.read_csv(input_path)
    input_rows = len(df)

    df["names"] = df["names"].map(normalize_text)
    df["orig_title"] = df["orig_title"].map(normalize_text)
    df["overview"] = df["overview"].map(normalize_text)
    df["status"] = df["status"].map(normalize_text)
    df["orig_lang"] = df["orig_lang"].map(normalize_text)
    df["country"] = df["country"].map(normalize_text)
    df["genre"] = df["genre"].map(normalize_genres)
    df["release_year"] = df["date_x"].map(extract_release_year)
    df = df.reset_index().rename(columns={"index": "source_row_id"})

    survivors = []
    unmatched = []
    seen_keys = {}
    released_count = 0

    for row in df.to_dict("records"):
        reason = None
        if row["status"].lower() != "released":
            reason = "Not released status."
        elif not row["names"]:
            reason = "Missing names."
        elif len(row["overview"]) < MIN_OVERVIEW_LEN:
            reason = "Missing overview."
        elif row["release_year"] is None:
            reason = "Missing release year."
        else:
            released_count += 1
            key = dedupe_key(row["names"], row["release_year"])
            if key in seen_keys:
                reason = "Duplicate normalized title + year."
            else:
                seen_keys[key] = row["source_row_id"]

        if reason:
            unmatched.append({
                "source_row_id": row["source_row_id"],
                "names": row["names"],
                "orig_title": row["orig_title"],
                "release_year": row["release_year"],
                "reason": reason,
                "unmatched_reason": reason,
                "best_candidate_title": None,
                "best_candidate_original_title": None,
                "best_candidate_year": None,
                "best_candidate_tmdb_id": None,
                "best_candidate_confidence": None,
                "duplicate_of_tmdb_id": None,
            })
        else:
            survivors.append(row)

    stats = {
        "input_rows": input_rows,
        "released_rows": released_count,
        "duplicates_removed": sum(1 for u in unmatched if u["reason"] == "Duplicate normalized title + year."),
        "rows_after_cleaning": len(survivors),
    }
    return survivors, unmatched, stats


def build_canonical(survivors: list, client: TmdbClient, limit: Optional[int], workers: int):
    rows = survivors[:limit] if limit else survivors
    canonical_rows = []
    unmatched_rows = []
    lock = Lock()

    def process(row):
        lang_code = normalize_language(row["orig_lang"])
        try:
            result = match_movie(client, row["names"], row["orig_title"], row["release_year"], lang_code, row["overview"])
        except RuntimeError:
            # TMDB request failed even after _get's internal retries - don't
            # let one bad row take down the whole build. Record it as
            # unmatched so it's visible and can be re-run later (the cache
            # means already-succeeded rows won't be re-fetched).
            return {"unmatched": {
                "source_row_id": row["source_row_id"],
                "names": row["names"],
                "orig_title": row["orig_title"],
                "release_year": row["release_year"],
                "reason": "TMDB request failed after retries.",
                "unmatched_reason": "TMDB request failed after retries.",
                "best_candidate_title": None,
                "best_candidate_original_title": None,
                "best_candidate_year": None,
                "best_candidate_tmdb_id": None,
                "best_candidate_confidence": None,
                "duplicate_of_tmdb_id": None,
            }}

        if result.matched:
            details = {}
            try:
                details = client.movie_details(result.tmdb_id)
            except Exception:
                details = {}

            cand = result.candidate or {}
            title = details.get("title") or cand.get("title") or ""
            original_title = details.get("original_title") or cand.get("original_title") or ""
            release_date = details.get("release_date") or cand.get("release_date") or ""
            release_year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else row["release_year"]

            tmdb_genres = ", ".join(g["name"] for g in details.get("genres", []) if g.get("name"))
            genres = tmdb_genres if tmdb_genres else row["genre"]

            overview = row["overview"] if len(row["overview"]) >= MIN_OVERVIEW_LEN else (details.get("overview") or cand.get("overview") or "")

            original_language = details.get("original_language") or cand.get("original_language") or lang_code or ""
            runtime = details.get("runtime")

            canonical = {
                "source_row_id": row["source_row_id"],
                "tmdb_id": result.tmdb_id,
                "title": title,
                "original_title": original_title,
                "release_date": release_date,
                "release_year": release_year,
                "genres": genres,
                "overview": overview,
                "score": row["score"],
                "original_language": original_language,
                "runtime": runtime,
                "source_name": row["names"],
                "source_original_title": row["orig_title"],
                "source_genre": row["genre"],
                "source_overview": row["overview"],
                "source_orig_lang": row["orig_lang"],
                "source_country": row["country"],
                "match_confidence": result.confidence,
                "match_reason": result.reason,
            }
            return {"canonical": canonical}
        else:
            unmatched = {
                "source_row_id": row["source_row_id"],
                "names": row["names"],
                "orig_title": row["orig_title"],
                "release_year": row["release_year"],
                "reason": result.reason,
                "unmatched_reason": result.reason,
                "best_candidate_title": result.best_candidate_title,
                "best_candidate_original_title": result.best_candidate_original_title,
                "best_candidate_year": result.best_candidate_year,
                "best_candidate_tmdb_id": result.best_candidate_tmdb_id,
                "best_candidate_confidence": result.best_candidate_confidence,
                "duplicate_of_tmdb_id": None,
            }
            return {"unmatched": unmatched}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, row) for row in rows]
        done = 0
        for fut in as_completed(futures):
            out = fut.result()
            with lock:
                if "unmatched" in out:
                    unmatched_rows.append(out["unmatched"])
                else:
                    canonical_rows.append(out["canonical"])
                done += 1
                if done % 200 == 0:
                    print(f"  ... processed {done}/{len(rows)}", file=sys.stderr)

    canonical_rows.sort(key=lambda r: r["source_row_id"])
    unmatched_rows.sort(key=lambda r: r["source_row_id"])
    return canonical_rows, unmatched_rows


def validate_outputs(canonical_path: Path, unmatched_path: Path, threshold: float) -> list:
    """Returns a list of error strings; empty list means all checks passed."""
    errors = []
    if not canonical_path.exists():
        errors.append(f"{canonical_path} does not exist.")
    if not unmatched_path.exists():
        errors.append(f"{unmatched_path} does not exist.")
    if errors:
        return errors

    canonical = pd.read_csv(canonical_path)
    unmatched = pd.read_csv(unmatched_path)

    missing_canonical_cols = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing_canonical_cols:
        errors.append(f"canonical_movies.csv missing columns: {missing_canonical_cols}")
    missing_unmatched_cols = [c for c in UNMATCHED_COLUMNS if c not in unmatched.columns]
    if missing_unmatched_cols:
        errors.append(f"unmatched_movies.csv missing columns: {missing_unmatched_cols}")
    if errors:
        return errors

    if len(canonical) > 0:
        if canonical["tmdb_id"].isna().any():
            errors.append("Some canonical_movies.csv rows have an empty tmdb_id.")
        dup_ids = canonical["tmdb_id"].value_counts()
        dup_ids = dup_ids[dup_ids > 1]
        if len(dup_ids) > 0:
            errors.append(f"Duplicate tmdb_id values found in canonical_movies.csv: {dup_ids.index.tolist()}")
        below_threshold = canonical[canonical["match_confidence"] < threshold]
        if len(below_threshold) > 0:
            errors.append(f"{len(below_threshold)} canonical rows have match_confidence below threshold {threshold}.")
        dedupe_keys = canonical.apply(lambda r: dedupe_key(r["source_name"], r["release_year"]), axis=1)
        if dedupe_keys.duplicated().any():
            errors.append("Duplicate normalized title+year found within canonical_movies.csv.")

    return errors


def print_review(unmatched_path: Path, sample_size: int = 20):
    if not unmatched_path.exists():
        print(f"No unmatched file found at {unmatched_path}. Run the script first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(unmatched_path)
    print(f"Total unmatched rows: {len(df)}\n")
    print("Reason breakdown:")
    for reason, count in Counter(df["reason"]).most_common():
        print(f"  {count:6d}  {reason}")
    print(f"\nSample of up to {sample_size} unmatched rows:")
    cols = ["source_row_id", "names", "release_year", "reason", "best_candidate_title", "best_candidate_confidence"]
    with pd.option_context("display.max_colwidth", 40, "display.width", 160):
        print(df[cols].head(sample_size).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--unmatched", default=DEFAULT_UNMATCHED_CSV)
    parser.add_argument("--report", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Only TMDB-match the first N cleaned rows (smoke test).")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--review-unmatched", action="store_true", help="Summarize an existing unmatched_movies.csv and exit.")
    args = parser.parse_args()

    if args.review_unmatched:
        print_review(Path(args.unmatched))
        return

    if not TMDB_API_KEY and not TMDB_API_READ_TOKEN:
        print(
            "ERROR: no TMDB credentials found. Set TMDB_API_KEY (v3 api key) or "
            "TMDB_API_READ_TOKEN (v4 bearer token) as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(args.input)
    survivors, cleaning_unmatched, clean_stats = load_and_triage(input_path)
    print(f"Loaded {clean_stats['input_rows']} rows from {input_path}")
    print(f"Released rows: {clean_stats['released_rows']}")
    print(f"Duplicate rows removed: {clean_stats['duplicates_removed']}")
    print(f"Rows after cleaning (ready for TMDB matching): {clean_stats['rows_after_cleaning']}")
    if args.limit:
        print(f"--limit set: only matching the first {args.limit} cleaned rows")

    client = TmdbClient(Path(args.cache), use_cache=not args.no_cache)
    try:
        canonical_rows, matching_unmatched = build_canonical(survivors, client, args.limit, args.workers)
    finally:
        client.save_cache()

    canonical_rows, dedup_demoted = deduplicate_by_tmdb_id(canonical_rows)
    if dedup_demoted:
        print(f"Deduplicated {len(dedup_demoted)} row(s) sharing a tmdb_id with a better-scoring row.")

    unmatched_rows = cleaning_unmatched + matching_unmatched + dedup_demoted
    unmatched_rows.sort(key=lambda r: r["source_row_id"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(canonical_rows, columns=CANONICAL_COLUMNS).to_csv(output_path, index=False)

    unmatched_path = Path(args.unmatched)
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unmatched_rows, columns=UNMATCHED_COLUMNS).to_csv(unmatched_path, index=False)

    matched_count = len(canonical_rows)
    unmatched_count = len(unmatched_rows)
    denom = matched_count + len(matching_unmatched) + len(dedup_demoted)  # match rate is over rows actually attempted against TMDB
    match_rate = (matched_count / denom * 100) if denom else 0.0

    report = {
        "input_rows": clean_stats["input_rows"],
        "rows_after_cleaning": clean_stats["rows_after_cleaning"],
        "released_rows": clean_stats["released_rows"],
        "duplicates_removed": clean_stats["duplicates_removed"],
        "matched_rows": matched_count,
        "unmatched_rows": unmatched_count,
        "match_rate": round(match_rate, 2),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "output_files": {
            "canonical_movies": str(output_path),
            "unmatched_movies": str(unmatched_path),
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"-> {output_path}")
    print(f"-> {report_path}")

    if match_rate < 40:
        print(f"WARNING: match rate ({match_rate:.2f}%) is suspiciously low.", file=sys.stderr)
    elif match_rate > 98:
        print(f"WARNING: match rate ({match_rate:.2f}%) is suspiciously high - double check the threshold isn't too lax.", file=sys.stderr)

    errors = validate_outputs(output_path, unmatched_path, CONFIDENCE_THRESHOLD)
    if errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(2)
    print("\nValidation checks passed.")


if __name__ == "__main__":
    main()
