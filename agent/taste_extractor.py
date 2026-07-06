"""Stage 0 — Taste Extractor.

One LLM call that parses a natural-language request into structured
UserPreferences. If the incoming prompt is already a JSON object matching the
schema, the LLM call is skipped entirely (structured-input bypass).
"""
from __future__ import annotations

import json
from typing import List, Optional

from pydantic import ValidationError

from agent.clients.llm_client import llm_client
from agent.models import Message, UserPreferences
from agent.parsing import parse_json_object
from agent.prompts import TASTE_EXTRACTOR_SYSTEM_PROMPT
from agent.steps import log_step

MODULE = "TasteExtractor"

# Fields that mark a payload as "already structured" preferences.
_PREF_KEYS = set(UserPreferences.model_fields.keys())


def _try_structured_bypass(prompt: str) -> Optional[UserPreferences]:
    """Return UserPreferences if the prompt is already structured JSON, else None."""
    try:
        data = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    # Consider it structured if it uses only known preference keys and has ≥1.
    if not (set(data.keys()) & _PREF_KEYS) or (set(data.keys()) - _PREF_KEYS):
        return None
    try:
        return UserPreferences(**data)
    except ValidationError:
        return None


def _build_user_prompt(
    prompt: str, conversation_history: Optional[List[Message]]
) -> str:
    if not conversation_history:
        return prompt
    history_lines = [
        f"{m.role}: {m.content}" for m in conversation_history
    ]
    return (
        "Conversation so far:\n"
        + "\n".join(history_lines)
        + "\n\nCurrent message:\n"
        + prompt
    )


async def taste_extractor(
    prompt: str,
    conversation_history: Optional[List[Message]],
    steps: List[dict],
) -> UserPreferences:
    # --- Structured-input bypass (0 LLM calls) -------------------------------
    bypass = _try_structured_bypass(prompt)
    if bypass is not None:
        log_step(
            steps,
            module=MODULE,
            system_prompt=None,
            user_prompt="[structured-input bypass] " + prompt,
            response=bypass.model_dump(),
        )
        return bypass

    # --- One LLM call --------------------------------------------------------
    user_prompt = _build_user_prompt(prompt, conversation_history)
    messages = [
        {"role": "system", "content": TASTE_EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prefs = await _call_and_parse(messages, retry=True)

    log_step(
        steps,
        module=MODULE,
        system_prompt=TASTE_EXTRACTOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response=prefs.model_dump(),
    )
    return prefs


async def _call_and_parse(messages: List[dict], retry: bool) -> UserPreferences:
    resp = await llm_client.chat(
        messages, response_format={"type": "json_object"}
    )
    content = resp.choices[0].message.content or ""
    try:
        data = parse_json_object(content)
        return UserPreferences(**data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        if not retry:
            raise
        # Nudge and retry once (§12).
        nudge_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON matching the "
                    "UserPreferences schema. Return valid JSON only."
                ),
            },
        ]
        _ = exc
        return await _call_and_parse(nudge_messages, retry=False)
