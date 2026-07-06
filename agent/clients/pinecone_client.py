"""Pinecone query helper (read-only at runtime).

The index is pre-populated by a separate ingestion project — this client only
queries it. A single `query` method returns matches with their metadata.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pinecone import Pinecone

from agent import config


class PineconeClient:
    def __init__(self) -> None:
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


pinecone_client = PineconeClient()
