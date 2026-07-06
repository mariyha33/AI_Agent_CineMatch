"""Deterministic orchestrator — plain Python control flow, no LLM routing.

Runs Stage 0 (taste extraction) then the bounded ReAct <-> Reflection loop,
handling clarification short-circuits, Pinecone-unreachable fallback, and a
best-effort compose when passes are exhausted.
"""
from __future__ import annotations

from typing import List, Optional

from agent import config
from agent.models import Message, ReactContext, ReActDraft, UserPreferences
from agent.react_agent import react_agent
from agent.reflection_agent import reflection_agent
from agent.taste_extractor import taste_extractor


async def run_pipeline(
    prompt: str,
    conversation_history: Optional[List[Message]],
    steps: List[dict],
) -> str:
    # --- Stage 0 — Taste Extraction -----------------------------------------
    preferences = await taste_extractor(prompt, conversation_history, steps)

    # --- Stage 1 + 2 — ReAct <-> Reflection loop ----------------------------
    context = ReactContext(preferences=preferences, feedback=None, use_fallback=False)

    draft: ReActDraft = ReActDraft()
    for _pass in range(1, config.MAX_PASSES + 1):
        draft = await _run_react_with_fallback(context, steps)

        # ReAct asked the user a question -> short-circuit.
        if draft.is_clarification:
            return draft.clarification_question or "Could you clarify your request?"

        verdict = await reflection_agent(draft, preferences, steps)

        if verdict.decision == "clarify":
            return verdict.question or "Could you clarify your request?"
        if verdict.decision == "approve":
            return verdict.final_response or _compose_best_effort(draft, preferences)

        # Reject -> feed critique back into the next ReAct pass.
        context.feedback = verdict.critique
        context.use_fallback = verdict.use_fallback

    # Exhausted passes -> best effort from the last draft.
    return _compose_best_effort(draft, preferences)


async def _run_react_with_fallback(
    context: ReactContext, steps: List[dict]
) -> ReActDraft:
    """Run ReAct; if Pinecone is unreachable, retry once forcing fallback."""
    try:
        return await react_agent(context, steps)
    except Exception:
        if context.use_fallback:
            raise  # already in fallback mode; surface the error
        context.use_fallback = True
        return await react_agent(context, steps)


def _compose_best_effort(draft: ReActDraft, preferences: UserPreferences) -> str:
    """Format whatever candidates we have when Reflection never approved."""
    if not draft.candidates:
        return (
            "I couldn't find a confident match for your request. Try loosening a "
            "constraint (platform, country, year, or rating) and asking again."
        )
    country = preferences.country or "your region"
    platform = preferences.platforms[0] if preferences.platforms else "a supported service"
    lines: List[str] = [
        "Here are my best guesses (availability not fully confirmed):",
        "",
    ]
    for c in draft.candidates[: config.MAX_CANDIDATES]:
        year = f" ({c.year})" if c.year else ""
        rationale = c.rationale or "Matches the taste you described."
        lines.append(f"**{c.title}{year}** — {rationale}")
        lines.append(f"↳ Possibly on {platform} — {country}")
        lines.append("")
    return "\n".join(lines).strip()
