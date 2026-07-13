"""Stage 2 — Reflection Agent.

Judges taste/novelty for a list of candidates whose availability has ALREADY
been verified deterministically by the orchestrator (see
agent/tools/verify_recommendation.py, called directly — no LLM round-trip).
This agent makes exactly one LLM call: no tool loop, since there's nothing left
for it to look up. On approve it composes the final user-facing response
itself — there is no separate composer stage.
"""
from __future__ import annotations

import json
from typing import List

from agent import config
from agent.clients.llm_client import llm_client
from agent.models import Candidate, ReflectionVerdict, UserPreferences
from agent.parsing import parse_json_object
from agent.prompts import build_reflection_system_prompt
from agent.steps import log_step

MODULE = "ReflectionAgent"


def _build_user_context(
    verified_candidates: List[dict],
    preferences: UserPreferences,
    approved: List[Candidate],
) -> str:
    parts = [
        "User preferences (JSON):\n"
        + json.dumps(preferences.model_dump(), ensure_ascii=False)
    ]
    if approved:
        parts.append(
            "Already approved in earlier passes (JSON) — keep these in the final "
            "response if you approve:\n"
            + json.dumps([c.model_dump() for c in approved], ensure_ascii=False)
        )
    parts.append(
        "New candidates to judge, already availability-verified (JSON):\n"
        + json.dumps(verified_candidates, ensure_ascii=False)
        + "\n\nReturn your verdict JSON now."
    )
    return "\n\n".join(parts)


def _parse_verdict(content: str) -> ReflectionVerdict:
    data = parse_json_object(content)
    return ReflectionVerdict(
        decision=data.get("decision", "reject"),
        final_response=data.get("final_response"),
        critique=data.get("critique"),
        use_fallback=bool(data.get("use_fallback", False)),
        question=data.get("question"),
        approved_ids=[int(i) for i in (data.get("approved_ids") or [])],
        rejected=data.get("rejected") or [],
    )


async def reflection_agent(
    verified_candidates: List[dict],
    preferences: UserPreferences,
    approved: List[Candidate],
    steps: List[dict],
    interactive: bool = False,
    is_final_pass: bool = False,
) -> ReflectionVerdict:
    system_prompt = build_reflection_system_prompt(
        interactive=interactive,
        is_final_pass=is_final_pass,
        min_candidates=config.MIN_CANDIDATES,
        max_candidates=config.MAX_CANDIDATES,
        already_approved_count=len(approved),
    )
    user_context = _build_user_context(verified_candidates, preferences, approved)

    resp = await llm_client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]
    )
    message = resp.choices[0].message

    log_step(
        steps,
        module=MODULE,
        system_prompt=system_prompt,
        user_prompt=user_context,
        response={"content": message.content},
    )

    return _parse_verdict(message.content or "{}")
