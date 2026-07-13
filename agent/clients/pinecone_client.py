"""Pinecone query/upsert helper.

The index is normally pre-populated by scripts/build_rag_index.py (or a
separate ingestion project) — this client queries it, and can also upsert to
it when explicitly asked to.

Connection is lazy: the Pinecone SDK is only imported and the client only
constructed on first real use (query/upsert). This lets the rest of the app
import cleanly when Pinecone isn't configured (e.g. RAG_BACKEND=local).
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from agent import config


class PineconeNotConfigured(RuntimeError):
    """Raised when a Pinecone operation is attempted without PINECONE_API_KEY."""


class PineconeClient:
    def __init__(self) -> None:
        # Deferred — no network/SDK work happens here.
        self._pc = None
        self._index = None

    def _ensure_connected(self) -> None:
        if self._index is not None:
            return
        if not config.PINECONE_API_KEY:
            raise PineconeNotConfigured(
                "PINECONE_API_KEY is not set. Set RAG_BACKEND=local to use the "
                "local JSONL fallback instead, or configure Pinecone in .env."
            )
        from pinecone import Pinecone  # imported lazily so it's only required

        self._pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self._index = self._pc.Index(config.PINECONE_INDEX_NAME)

    def query(
        self,
        vector: List[float],
        filter: Optional[dict] = None,
        top_k: int = 10,
    ) -> List[dict]:
        """Query the index and return matches as plain dicts.

        Each returned dict has: id, score, and metadata (title, year, tmdb_id,
        genres, score, overview, ...).
        """
        self._ensure_connected()
        result = self._index.query(
            vector=vector,
            filter=filter or None,
            top_k=top_k,
            include_metadata=True,
        )
        matches = result.get("matches", []) if isinstance(result, dict) else result.matches
        out: List[dict] = []
        for m in matches:
            # Support both dict-like and attribute-like match objects.
            if isinstance(m, dict):
                out.append(
                    {
                        "id": m.get("id"),
                        "score": m.get("score"),
                        "metadata": m.get("metadata", {}) or {},
                    }
                )
            else:
                out.append(
                    {
                        "id": getattr(m, "id", None),
                        "score": getattr(m, "score", None),
                        "metadata": getattr(m, "metadata", {}) or {},
                    }
                )
        return out

    def upsert(self, vectors: Iterable[dict], batch_size: int = 100) -> int:
        """Upsert vectors into the index.

        Each item in `vectors` must be a dict with keys: id, values (the
        embedding), and metadata. Used by scripts/build_rag_index.py to
        populate the index with deterministic ids ("movie-{tmdb_id}").
        Returns the total number of vectors upserted.
        """
        self._ensure_connected()
        batch: List[dict] = []
        total = 0
        for item in vectors:
            batch.append(item)
            if len(batch) >= batch_size:
                self._index.upsert(vectors=batch)
                total += len(batch)
                batch = []
        if batch:
            self._index.upsert(vectors=batch)
            total += len(batch)
        return total


# Module-level singleton — cheap now (no connection happens until first
# query()/upsert() call), so it's safe to import even without Pinecone
# configured.
pinecone_client = PineconeClient()
