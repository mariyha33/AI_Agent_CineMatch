"""Shared helpers for coercing LLM text output into JSON."""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Remove surrounding ```json ... ``` markdown fences if present."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return text


def parse_json_object(text: str) -> Any:
    """Parse a JSON object from LLM text, tolerating markdown fences.

    Falls back to extracting the first {...} span if there is surrounding prose.
    Raises json.JSONDecodeError if nothing parses.
    """
    cleaned = strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise
