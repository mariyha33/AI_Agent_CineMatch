"""Thin async wrapper around the OpenAI SDK pointed at LLMod.ai.

Exposes two methods used across the pipeline:
  * chat(messages, tools=None, response_format=None) -> ChatCompletion
  * embed(text) -> list[float]

Both retry once on transient errors (timeout / 5xx) per the error-handling spec.

The underlying AsyncOpenAI client is constructed lazily — only on the first
real chat()/embed() call — so importing this module (directly, or transitively
via agent.tools.rag_search) never requires LLMOD_API_KEY. Workflows that don't
need real LLM calls (e.g. scripts/build_rag_index.py --mock-embeddings) never
hit this path at all.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from openai import AsyncOpenAI
from openai import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError

from agent import config

# Importing these exception classes has no side effects (no network, no
# credentials needed) — only constructing AsyncOpenAI does, so that part is
# deferred below.
_TRANSIENT_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)


class LLMClientNotConfigured(RuntimeError):
    """Raised when chat()/embed() is called without LLMOD_API_KEY set."""


class LLMClient:
    def __init__(self) -> None:
        # Deferred — no SDK/network work happens here.
        self._client = None

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        if not config.LLMOD_API_KEY:
            raise LLMClientNotConfigured(
                "LLMOD_API_KEY is not set. Set it in .env to make real chat/embedding "
                "calls. Mock-embedding workflows (e.g. scripts/build_rag_index.py "
                "--mock-embeddings) don't need it."
            )
        self._client = AsyncOpenAI(
            api_key=config.LLMOD_API_KEY,
            base_url=config.LLMOD_BASE_URL,
        )
        return self._client

    async def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        response_format: Optional[dict] = None,
    ) -> Any:
        """Call the text model. Returns the raw ChatCompletion object."""
        client = self._ensure_client()
        kwargs: dict = {"model": config.LLMOD_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        return await self._with_retry(
            lambda: client.chat.completions.create(**kwargs)
        )

    async def embed(self, text: str) -> List[float]:
        """Embed a single string, returning its vector."""
        client = self._ensure_client()
        resp = await self._with_retry(
            lambda: client.embeddings.create(
                model=config.LLMOD_EMBEDDING_MODEL, input=text
            )
        )
        return resp.data[0].embedding

    async def _with_retry(self, call):
        """Run an async-returning callable, retrying once on transient errors."""
        try:
            return await call()
        except _TRANSIENT_ERRORS:
            await asyncio.sleep(0.5)
            return await call()


# Module-level singleton — cheap now (no connection happens until first
# chat()/embed() call), so it's safe to import even without LLMOD_API_KEY set.
llm_client = LLMClient()
