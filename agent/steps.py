"""Shared helper for appending trace steps.

Every LLM call and every tool execution appends one step object to the shared
`steps` list (passed by reference through the whole pipeline). The schema is
fixed by the project spec (§13):

    {
      "module": "...",
      "prompt": {"system_prompt": ... , "user_prompt": ...},
      "response": {...}
    }
"""
from __future__ import annotations

from typing import Any, List, Optional


def log_step(
    steps: List[dict],
    module: str,
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    response: Any,
) -> None:
    steps.append(
        {
            "module": module,
            "prompt": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            "response": response,
        }
    )
