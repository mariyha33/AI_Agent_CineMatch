"""Async TMDB API helper (httpx).

No search-by-title endpoint is exposed - every candidate already carries a
tmdb_id (from RAG metadata or from discover), so title/year resolution is never
needed at runtime.

TMDB_API_KEY is optional at import time (see agent/config.py) - it is only
checked when a real API call is attempted, so importing this module never
requires TMDB_API_KEY. Local/mock RAG commands never call this client at all.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

from agent import config


class TMDBClientNotConfigured(RuntimeError):
    """Raised when a TMDB API call is attempted without TMDB_API_KEY set."""


class TMDBClient:
    def __init__(self) -> None:
        self._base_url = config.TMDB_BASE_URL
        self._api_key = config.TMDB_API_KEY

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._base_url, timeout=15.0)

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        if not self._api_key:
            raise TMDBClientNotConfigured(
                "TMDB_API_KEY is not set. Set it in .env to make real TMDB API "
                "calls. Local/mock RAG commands don't need it."
            )
        params = dict(params or {})
        params["api_key"] = self._api_key
        async with self._client() as client:
            resp = await client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

    async def get_movie(self, tmdb_id: int) -> dict:
        """Movie detail: genres, overview, rating, popularity, etc."""
        return await self._get(f"/movie/{tmdb_id}")

    async def get_watch_providers(self, tmdb_id: int) -> dict:
        """Availability by country: /movie/{id}/watch/providers."""
        return await self._get(f"/movie/{tmdb_id}/watch/providers")

    async def get_movie_keywords(self, tmdb_id: int) -> dict:
        """Keyword tags for a movie: /movie/{id}/keywords."""
        return await self._get(f"/movie/{tmdb_id}/keywords")

    async def discover_movies(self, filters: dict) -> dict:
        """Structured discovery: /discover/movie with the given query params."""
        return await self._get("/discover/movie", params=filters)


tmdb_client = TMDBClient()
