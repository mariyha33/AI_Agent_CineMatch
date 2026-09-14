"""build_rag_index.py — build the RAG document set for CineMatch.

Reads ONLY data/processed/canonical_movies.csv (never unmatched_movies.csv —
those rows have no verified tmdb_id and must never be recommended). For every
row it optionally enriches metadata (decade/director/keywords/vote stats —
see enrich_rows/--enrich-tmdb below), builds a text description, embeds it via
LLMod.ai (LLMOD_EMBEDDING_MODEL), and writes one JSON object per line to
data/processed/rag_documents.jsonl (or RAG_DOCUMENTS_PATH).

This file is what agent/clients/local_rag_client.py searches when
RAG_BACKEND=local (the default — no Pinecone account needed). Pass
--upsert-pinecone to also push the same vectors into Pinecone with
deterministic ids ("movie-{tmdb_id}"), for RAG_BACKEND=pinecone.

Usage:
    python scripts/build_rag_index.py                      # full build, local only
    python scripts/build_rag_index.py --limit 20            # smoke test, no cost
    python scripts/build_rag_index.py --limit 20 --mock-embeddings  # zero API calls
    python scripts/build_rag_index.py --upsert-pinecone      # also push to Pinecone

Metadata enrichment (decade/director/keywords/vote_count/vote_average — see
Phase 2, agent/tools/rag_search.py):
    python scripts/build_rag_index.py --limit 20 --enrich-tmdb            # real TMDB calls, needs TMDB_API_KEY
    python scripts/build_rag_index.py --limit 20 --enrich-tmdb --mock-tmdb-enrich  # zero API calls, deterministic fixture data
Enrichment is opt-in and additive: without --enrich-tmdb, rows simply carry
director=None/keywords=[]/vote_count=None/vote_average=None, and the rest of
the pipeline (embedding text, storage, rag_search) tolerates that fine. A row
that already carries director/keywords (e.g. from an already-enriched source)
is never re-fetched.

Safe to re-run: existing tmdb_ids already present in the output file are
skipped by default (use --overwrite to rebuild from scratch). Never modifies
Phase 1 files or data/cache.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
from typing import Any, Iterable, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent import config  # noqa: E402

DEFAULT_INPUT = os.path.join(_ROOT, "data", "processed", "canonical_movies.csv")

# Hard cap on how many keyword/theme tags a single movie can carry into
# embedding text and stored metadata — applied everywhere a keywords list is
# produced or read (TMDB fetch, CSV load, and defensively again right before
# embedding), so one unusually keyword-heavy movie can never blow up
# embedding text size or per-document metadata size.
MAX_KEYWORDS = 15


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_ROOT, path)


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_list(value: Any) -> List[str]:
    if not value:
        return []
    return [g.strip() for g in str(value).split(",") if g.strip()]


# Kept as a separate name for readability at call sites — same comma-split
# logic as _parse_list.
_parse_genres = _parse_list


def _decade_label(year: Optional[int]) -> Optional[str]:
    """e.g. 1994 -> '1990s'. None in, None out — never guess a decade for a
    movie with no known release year."""
    if year is None:
        return None
    return f"{(year // 10) * 10}s"


def load_canonical_rows(input_csv: str) -> List[dict]:
    """Load canonical_movies.csv and build one RAG-ready record per valid row.

    Enforces the project invariant here too: a row with no usable tmdb_id is
    skipped (it should never happen in canonical_movies.csv, but we don't
    trust blindly — see the Phase 1 "14 missing rows" incident).

    director/keywords/vote_count/vote_average/decade are read from the CSV
    only if those columns exist (they don't yet — see Phase 2 enrichment,
    enrich_rows below, which fills them in-memory for rows that lack them).
    A CSV without these columns loads exactly as before.
    """
    rows: List[dict] = []
    skipped_no_tmdb_id = 0
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            tmdb_id = _to_int(raw.get("tmdb_id"))
            if tmdb_id is None:
                skipped_no_tmdb_id += 1
                continue
            title = (raw.get("title") or "").strip()
            year = _to_int(raw.get("release_year"))
            genres = _parse_genres(raw.get("genres"))
            overview = (raw.get("overview") or "").strip()
            score = _to_float(raw.get("score"))
            runtime = _to_int(raw.get("runtime"))
            director = (raw.get("director") or "").strip() or None
            keywords = _parse_list(raw.get("keywords"))[:MAX_KEYWORDS]
            vote_count = _to_int(raw.get("vote_count"))
            vote_average = _to_float(raw.get("vote_average"))
            decade = (raw.get("decade") or "").strip() or None
            rows.append(
                {
                    "tmdb_id": tmdb_id,
                    "title": title,
                    "year": year,
                    "genres": genres,
                    "overview": overview,
                    "score": score,
                    "runtime": runtime,
                    "director": director,
                    "keywords": keywords,
                    "vote_count": vote_count,
                    "vote_average": vote_average,
                    "decade": decade,
                }
            )
    if skipped_no_tmdb_id:
        print(
            f"WARNING: skipped {skipped_no_tmdb_id} row(s) with no tmdb_id in "
            f"{input_csv} — this should be 0 for canonical_movies.csv.",
            file=sys.stderr,
        )
    return rows


def build_embedding_text(row: dict) -> str:
    """Structured, labeled embedding text (Phase 2). Every field falls back to
    a plain "unknown" placeholder rather than raising or omitting the line
    entirely — a row missing director/keywords/decade (the common case until
    a movie has been enriched — see enrich_rows) must still embed cleanly.
    Deliberately excludes source_country and any availability info; this text
    is content/taste-only, mirroring the same isolation principle as Phase 1C.
    """
    title = row.get("title") or "Unknown title"
    year = row.get("year")
    year_text = str(year) if year is not None else "Unknown year"
    decade = row.get("decade") or _decade_label(year) or "Unknown era"
    genres_text = ", ".join(row.get("genres") or []) or "Unknown"
    director = row.get("director") or "Unknown"
    # Defensively capped again here (belt-and-suspenders on top of the caps
    # already applied in _fetch_one_enrichment / load_canonical_rows) so this
    # function alone can never produce runaway embedding text regardless of
    # what a future caller passes in.
    keywords = (row.get("keywords") or [])[:MAX_KEYWORDS]
    keywords_text = ", ".join(keywords) if keywords else "None listed"
    overview = (row.get("overview") or "").strip() or "No overview available."
    return (
        f"Title: {title}\n"
        f"Year: {year_text}\n"
        f"Era: {decade}\n"
        f"Genres: {genres_text}\n"
        f"Director: {director}\n"
        f"Keywords/Themes: {keywords_text}\n"
        f"Overview: {overview}"
    ).strip()


def _mock_embedding(text: str, dim: int = 32) -> List[float]:
    """Deterministic pseudo-embedding for --mock-embeddings smoke testing.

    Not semantically meaningful — only used so the whole pipeline (write,
    read, cosine similarity, filtering) can be exercised with zero API calls
    and zero cost.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        vals.append((byte / 255.0) * 2 - 1)
    return vals


def load_existing_tmdb_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    ids = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if doc.get("tmdb_id") is not None:
                ids.add(doc["tmdb_id"])
    return ids


async def _fetch_one_enrichment(tmdb_id: int) -> dict:
    """Fetch director + keywords + vote stats for one movie, reusing the
    existing async TMDB client methods (agent/clients/tmdb_client.py) —
    get_movie, get_movie_keywords, get_movie_credits — the same three methods
    agent/tools/verify_recommendation.py already calls per-candidate at
    request time. Never raises: any TMDB failure for a single row degrades to
    "no enrichment for this row" rather than aborting the whole batch.
    """
    from agent.clients.tmdb_client import tmdb_client  # noqa: E402 (needs TMDB_API_KEY)

    try:
        movie, keywords_payload, credits = await asyncio.gather(
            tmdb_client.get_movie(tmdb_id),
            tmdb_client.get_movie_keywords(tmdb_id),
            tmdb_client.get_movie_credits(tmdb_id),
        )
    except Exception as exc:
        print(f"  WARNING: enrichment failed for tmdb_id={tmdb_id}: {exc}", file=sys.stderr)
        return {}

    director = None
    for c in credits.get("crew") or []:
        if (c.get("job") or "").strip().lower() == "director":
            director = (c.get("name") or "").strip() or None
            break

    keywords = [
        k.get("name") for k in (keywords_payload.get("keywords") or []) if k.get("name")
    ][:MAX_KEYWORDS]

    return {
        "director": director,
        "keywords": keywords,
        "vote_count": movie.get("vote_count"),
        "vote_average": movie.get("vote_average"),
    }


def _mock_enrichment(row: dict) -> dict:
    """Deterministic fixture for --mock-tmdb-enrich, so the enrichment code
    path (including build_embedding_text's use of the new fields) can be
    exercised with zero TMDB calls — mirrors --mock-embeddings."""
    seed = row.get("tmdb_id") or 0
    return {
        "director": f"Mock Director {seed % 7}",
        "keywords": [f"mock-keyword-{seed % 5}", f"mock-keyword-{(seed + 1) % 5}"],
        "vote_count": 100 + (seed % 5000),
        "vote_average": round(5.0 + (seed % 50) / 10.0, 1),
    }


async def enrich_rows(
    rows: List[dict],
    concurrency: int,
    enrich: bool,
    mock: bool,
) -> List[dict]:
    """Fill in decade (always, free — derived from year) and, only if
    requested, director/keywords/vote_count/vote_average via TMDB.

    Additive and idempotent: a row that already carries director/keywords
    (e.g. loaded from an already-enriched source) is left untouched rather
    than re-fetched. Without --enrich-tmdb this never touches the network —
    rows simply keep director=None/keywords=[]/vote_count=None/
    vote_average=None, which build_embedding_text and rag_search both handle
    without error.
    """
    for row in rows:
        row.setdefault("decade", _decade_label(row.get("year")))

    if not enrich:
        for row in rows:
            row.setdefault("director", None)
            row.setdefault("keywords", [])
            row.setdefault("vote_count", None)
            row.setdefault("vote_average", None)
        return rows

    semaphore = asyncio.Semaphore(concurrency)

    async def _enrich_one(row: dict) -> None:
        if row.get("director") or row.get("keywords"):
            return  # already enriched (e.g. present in the source CSV)
        async with semaphore:
            try:
                data = _mock_enrichment(row) if mock else await _fetch_one_enrichment(row["tmdb_id"])
            except Exception as exc:
                # Extra safety net: _fetch_one_enrichment already catches TMDB
                # errors internally and returns {} rather than raising, but
                # this guarantees that even an unexpected bug here can never
                # abort asyncio.gather() for the WHOLE batch below — one bad
                # row must never cost the other ~9k rows their enrichment.
                print(
                    f"  WARNING: enrichment crashed for tmdb_id={row.get('tmdb_id')}: {exc}",
                    file=sys.stderr,
                )
                data = {}
            row["director"] = data.get("director")
            row["keywords"] = (data.get("keywords") or [])[:MAX_KEYWORDS]
            row["vote_count"] = data.get("vote_count")
            row["vote_average"] = data.get("vote_average")

    await asyncio.gather(*(_enrich_one(r) for r in rows))
    return rows


async def embed_rows(
    rows: List[dict],
    concurrency: int,
    mock: bool,
) -> List[dict]:
    """Embed each row's text, returning rows with an added 'embedding' field."""
    if mock:
        for row in rows:
            row["embedding"] = _mock_embedding(build_embedding_text(row))
        return rows

    from agent.clients.llm_client import llm_client  # noqa: E402 (needs LLMOD_API_KEY)

    semaphore = asyncio.Semaphore(concurrency)

    async def _embed_one(row: dict) -> dict:
        async with semaphore:
            text = build_embedding_text(row)
            row["embedding"] = await llm_client.embed(text)
            return row

    done = 0
    total = len(rows)

    async def _embed_with_progress(row: dict) -> dict:
        nonlocal done
        result = await _embed_one(row)
        done += 1
        if done % 100 == 0 or done == total:
            print(f"  embedded {done}/{total}", file=sys.stderr)
        return result

    return await asyncio.gather(*(_embed_with_progress(r) for r in rows))


def write_jsonl(rows: Iterable[dict], output_path: str, append: bool) -> int:
    """Write rows as JSONL, atomically.

    Writes to a temp file in the same directory and then os.replace()s it into
    place, so a crash mid-write never leaves a truncated/corrupt
    rag_documents.jsonl. When append=True, existing lines are copied into the
    temp file first, so the final replace is still a single atomic swap.
    """
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f".{os.path.basename(output_path)}.tmp")

    new_count = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        if append and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as existing:
                for line in existing:
                    f.write(line if line.endswith("\n") else line + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            new_count += 1
    os.replace(tmp_path, output_path)
    return new_count


def upsert_to_pinecone(rows: List[dict], batch_size: int = 100) -> int:
    from agent.clients.pinecone_client import pinecone_client  # noqa: E402

    def _vectors():
        for row in rows:
            yield {
                "id": f"movie-{row['tmdb_id']}",
                "values": row["embedding"],
                "metadata": {
                    "title": row["title"],
                    "year": row["year"],
                    "tmdb_id": row["tmdb_id"],
                    "genres": row["genres"],
                    "score": row["score"],
                    "overview": row["overview"],
                    "runtime": row["runtime"],
                    "decade": row.get("decade"),
                    "director": row.get("director"),
                    "keywords": row.get("keywords") or [],
                    "vote_count": row.get("vote_count"),
                    "vote_average": row.get("vote_average"),
                },
            }

    return pinecone_client.upsert(_vectors(), batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to canonical_movies.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write rag_documents.jsonl (default: RAG_DOCUMENTS_PATH from config)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke testing)")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent embedding calls")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild from scratch instead of skipping tmdb_ids already in the output file",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic fake embeddings instead of calling the LLM (zero cost, for pipeline smoke tests)",
    )
    parser.add_argument(
        "--upsert-pinecone",
        action="store_true",
        help="Also upsert the built vectors into Pinecone (requires PINECONE_API_KEY)",
    )
    parser.add_argument(
        "--enrich-tmdb",
        action="store_true",
        help="Fetch director/keywords/vote_count/vote_average per row via TMDB (requires TMDB_API_KEY). Without this flag, those fields are simply left empty.",
    )
    parser.add_argument(
        "--mock-tmdb-enrich",
        action="store_true",
        help="With --enrich-tmdb, use deterministic fixture data instead of calling TMDB (zero cost, for pipeline smoke tests)",
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output or config.RAG_DOCUMENTS_PATH)

    print(f"Loading canonical movies from {args.input} ...")
    rows = load_canonical_rows(args.input)
    print(f"  {len(rows)} rows with a valid tmdb_id.")

    if not args.overwrite:
        existing_ids = load_existing_tmdb_ids(output_path)
        if existing_ids:
            before = len(rows)
            rows = [r for r in rows if r["tmdb_id"] not in existing_ids]
            print(f"  skipping {before - len(rows)} row(s) already in {output_path}")

    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"  --limit applied: processing {len(rows)} row(s)")

    if not rows:
        print("Nothing to embed (all rows already indexed, or --limit 0). Done.")
        return

    print(f"Enriching {len(rows)} row(s) (enrich_tmdb={args.enrich_tmdb}, mock={args.mock_tmdb_enrich}) ...")
    rows = asyncio.run(
        enrich_rows(rows, concurrency=args.concurrency, enrich=args.enrich_tmdb, mock=args.mock_tmdb_enrich)
    )

    print(f"Embedding {len(rows)} row(s) (mock={args.mock_embeddings}) ...")
    rows = asyncio.run(embed_rows(rows, concurrency=args.concurrency, mock=args.mock_embeddings))

    append = not args.overwrite and os.path.exists(output_path)
    written = write_jsonl(rows, output_path, append=append)
    print(f"Wrote {written} document(s) to {output_path} (append={append}).")

    if args.upsert_pinecone:
        print("Upserting to Pinecone ...")
        n = upsert_to_pinecone(rows)
        print(f"Upserted {n} vector(s) to Pinecone index '{config.PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()
