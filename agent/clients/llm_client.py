"""Thin async wrapper around the OpenAI SDK pointed at LLMod.ai.

Exposes two methods used across the pipeline:
  * chat(messages, tools=None, response_format=None) -> ChatCompletion
  * embed(text) -> list[float]

Both retry once on transient errors (timeout / 5xx) per the error-handling spec.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from openai import AsyncOpenAI
from openai import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError

from agent import config

_TRANSIENT_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=config.LLMOD_API_KEY,
            base_url=config.LLMOD_BASE_URL,
        )

    async def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        response_format: Optional[dict] = None,
    ) -> Any:
        """Call the text model. Returns the raw ChatCompletion object."""
        kwargs: dict = {"model": config.LLMOD_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        return await self._with_retry(
            lambda: self._client.chat.completions.create(**kwargs)
        )

    async def embed(self, text: str) -> List[float]:
        """Embed a single string, returning its vector."""
        resp = await self._with_retry(
            lambda: self._client.embeddings.create(
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


# Module-level singleton — clients are cheap and connection-pooled.
llm_client = LLMClient()
