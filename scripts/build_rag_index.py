"""build_rag_index.py — build the RAG document set for CineMatch.

Reads ONLY data/processed/canonical_movies.csv (never unmatched_movies.csv —
those rows have no verified tmdb_id and must never be recommended). For every
row it builds a short text description, embeds it via LLMod.ai
(LLMOD_EMBEDDING_MODEL), and writes one JSON object per line to
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
    python scripts/build_rag_index.py --upsert-existing-jsonl  # push an already-built
                                                                  # jsonl to Pinecone,
                                                                  # zero LLMOD calls

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


def _parse_genres(value: Any) -> List[str]:
    if not value:
        return []
    return [g.strip() for g in str(value).split(",") if g.strip()]


def load_canonical_rows(input_csv: str) -> List[dict]:
    """Load canonical_movies.csv and build one RAG-ready record per valid row.

    Enforces the project invariant here too: a row with no usable tmdb_id is
    skipped (it should never happen in canonical_movies.csv, but we don't
    trust blindly — see the Phase 1 "14 missing rows" incident).
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
            rows.append(
                {
                    "tmdb_id": tmdb_id,
                    "title": title,
                    "year": year,
                    "genres": genres,
                    "overview": overview,
                    "score": score,
                    "runtime": runtime,
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
    genres_text = ", ".join(row["genres"]) if row["genres"] else "Unknown"
    year_text = row["year"] if row["year"] is not None else "Unknown year"
    return (
        f"{row['title']} ({year_text}). Genres: {genres_text}. {row['overview']}"
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
                },
            }

    return pinecone_client.upsert(_vectors(), batch_size=batch_size)


def load_all_documents(path: str) -> List[dict]:
    """Read every JSON object from an existing rag_documents.jsonl-style file.

    Malformed lines are skipped with a warning rather than aborting the whole
    read, since this is meant to run against a file that may have been built
    over several sessions.
    """
    docs: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed line {line_no}: {exc}", file=sys.stderr)
    return docs


def upsert_existing_jsonl(
    path: str,
    limit: Optional[int] = None,
    batch_size: int = 100,
) -> int:
    """Upsert vectors already present in an existing rag_documents.jsonl to
    Pinecone, without re-embedding anything.

    - Makes zero LLMOD calls (embeddings are read as-is from the file).
    - Never writes to rag_documents.jsonl (read-only here).
    - Never touches Phase 1 files (canonical_movies.csv / unmatched_movies.csv
      / canonical_build_report.json) — this function only reads the RAG jsonl
      and calls Pinecone.
    - Enforces the same "no tmdb_id -> no embedding -> no upsert" invariant
      used everywhere else: any row missing tmdb_id or embedding is skipped.
    """
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Build it first with scripts/build_rag_index.py.", file=sys.stderr)
        return 0

    docs = load_all_documents(path)
    print(f"Read {len(docs)} row(s) from {path}.")

    valid_rows: List[dict] = []
    skipped = 0
    for doc in docs:
        if doc.get("tmdb_id") is None or not doc.get("embedding"):
            skipped += 1
            continue
        valid_rows.append(doc)

    if skipped:
        print(
            f"WARNING: skipped {skipped} row(s) missing tmdb_id and/or embedding "
            "(not upserted).",
            file=sys.stderr,
        )

    if limit is not None:
        valid_rows = valid_rows[:limit]
        print(f"  --limit applied: upserting {len(valid_rows)} row(s)")

    if not valid_rows:
        print("Nothing valid to upsert. Done.")
        return 0

    print(
        f"Upserting {len(valid_rows)} vector(s) to Pinecone "
        "(no LLMOD calls, rag_documents.jsonl not modified) ..."
    )
    n = upsert_to_pinecone(valid_rows, batch_size=batch_size)
    print(f"Upserted {n} vector(s) to Pinecone index '{config.PINECONE_INDEX_NAME}'.")
    return n


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
        "--upsert-existing-jsonl",
        action="store_true",
        help=(
            "Upsert vectors already in rag_documents.jsonl to Pinecone without "
            "re-embedding or calling LLMOD. Skips the canonical-CSV/embedding "
            "pipeline entirely; does not modify rag_documents.jsonl or any "
            "Phase 1 file. Combine with --limit to test on a small batch first."
        ),
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output or config.RAG_DOCUMENTS_PATH)

    if args.upsert_existing_jsonl:
        if args.upsert_pinecone:
            print(
                "Note: --upsert-existing-jsonl and --upsert-pinecone both set; "
                "running --upsert-existing-jsonl only (no canonical-CSV build).",
                file=sys.stderr,
            )
        upsert_existing_jsonl(output_path, limit=args.limit)
        return

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
