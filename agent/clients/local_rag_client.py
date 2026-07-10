"""Local JSONL-based vector search — a zero-dependency fallback for rag_search
when Pinecone isn't configured (RAG_BACKEND=local, the default).

Reads data/processed/rag_documents.jsonl, built by scripts/build_rag_index.py
from data/processed/canonical_movies.csv ONLY (never unmatched_movies.csv), so
every document here is guaranteed to carry a valid tmdb_id.

Documents and their embeddings are cached in memory after the first load, since
the file doesn't change during a single process's lifetime.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, List, Optional

from agent import config


class LocalRagNotBuilt(RuntimeError):
    """Raised when rag_documents.jsonl doesn't exist yet."""


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class LocalRagClient:
    def __init__(self) -> None:
        self._docs: Optional[List[dict]] = None
        self._path: Optional[str] = None

    def _resolve_path(self) -> str:
        path = config.RAG_DOCUMENTS_PATH
        if not os.path.isabs(path):
            # Resolve relative to the project root (two levels up from this file).
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(root, path)
        return path

    def _load(self) -> List[dict]:
        path = self._resolve_path()
        if self._docs is not None and self._path == path:
            return self._docs
        if not os.path.exists(path):
            raise LocalRagNotBuilt(
                f"{path} not found. Run `python scripts/build_rag_index.py` first "
                "to build the local RAG document set from canonical_movies.csv."
            )
        docs: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                # Enforce the invariant even at read time: no tmdb_id, no doc.
                if doc.get("tmdb_id") is not None and doc.get("embedding"):
                    docs.append(doc)
        self._docs = docs
        self._path = path
        return docs

    def search(
        self,
        vector: List[float],
        filter: Optional[dict] = None,
        top_k: int = 10,
    ) -> List[dict]:
        """Return the top_k most similar documents to `vector`.

        `filter` mirrors the subset of the Pinecone filter shape used by
        rag_search: {"genres": {"$in": [...]}, "year": {"$gte":, "$lte":},
        "score": {"$gte":}}.
        """
        docs = self._load()
        genres_filter = None
        year_gte = year_lte = None
        score_gte = None
        if filter:
            if "genres" in filter and "$in" in filter["genres"]:
                genres_filter = set(filter["genres"]["$in"])
            if "year" in filter:
                year_gte = filter["year"].get("$gte")
                year_lte = filter["year"].get("$lte")
            if "score" in filter:
                score_gte = filter["score"].get("$gte")

        scored: List[tuple] = []
        for doc in docs:
            if genres_filter and not (genres_filter & set(doc.get("genres") or [])):
                continue
            year = doc.get("year")
            if year_gte is not None and (year is None or year < year_gte):
                continue
            if year_lte is not None and (year is None or year > year_lte):
                continue
            if score_gte is not None and (doc.get("score") is None or doc.get("score") < score_gte):
                continue
            sim = _cosine_similarity(vector, doc["embedding"])
            scored.append((sim, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:top_k]

        out: List[dict] = []
        for sim, doc in top:
            out.append(
                {
                    "id": f"movie-{doc['tmdb_id']}",
                    "score": sim,
                    "metadata": {
                        "title": doc.get("title"),
                        "year": doc.get("year"),
                        "tmdb_id": doc.get("tmdb_id"),
                        "genres": doc.get("genres", []),
                        "score": doc.get("score"),
                        "overview": doc.get("overview"),
                        "runtime": doc.get("runtime"),
                    },
                }
            )
        return out


# Module-level singleton — loading is deferred until the first search() call.
local_rag_client = LocalRagClient()
