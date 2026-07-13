"""Stage 1 — ReAct Agent.

A true agentic loop using the LLM's native function-calling. The LLM decides
which tool to call, in what order, and when to stop. When it stops (returns text
with no tool calls) it must emit a JSON ReActDraft.
"""
from __future__ import annotations

import asyncio
import json
from typing import List, Optional, Tuple

from agent import config
from agent.clients.llm_client import llm_client
from agent.models import Candidate, ReactContext, ReActDraft
from agent.parsing import parse_json_object
from agent.prompts import build_react_system_prompt
from agent.steps import log_step
from agent.tools import ask_user_clarification, registry
from agent.tools.rag_search import RagUnavailable

MODULE = "ReActAgent"


def _rag_budget_for_pass(pass_number: int) -> int:
    """Max rag_search calls allowed this pass; -1 means unlimited (see
    config.RAG_BUDGET_BY_PASS). The RAG index carries no availability signal,
    so later passes are steered toward tmdb_fallback_search instead."""
    budgets = config.RAG_BUDGET_BY_PASS
    idx = min(max(pass_number - 1, 0), len(budgets) - 1)
    return budgets[idx]


def _allowed_tools(context: ReactContext, rag_remaining: bool) -> List[str]:
    """rag_search while its per-pass budget isn't exhausted; fallback once
    enabled (always true from pass 2 on — see orchestrator); clarification
    only interactively (and never on the final pass — there is no more
    retrying to clarify into)."""
    tools: List[str] = []
    if rag_remaining:
        tools.append("rag_search")
    if context.use_fallback or context.pass_number >= 2:
        tools.append("tmdb_fallback_search")
    if context.interactive and not context.is_final_pass:
        tools.append("ask_user_clarification")
    return tools


def _build_user_context(context: ReactContext) -> str:
    prefs = context.preferences
    needed = context.remaining_needed or config.MIN_CANDIDATES
    return (
        "User preferences (JSON):\n"
        + json.dumps(prefs.model_dump(), ensure_ascii=False)
        + f"\n\nFind at least {needed} NEW matching movie(s) (a couple extra as "
        + "buffer is fine in case some don't pass review)."
    )


def _assistant_message(message) -> dict:
    msg: dict = {"role": "assistant", "content": message.content or None}
    if message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return msg


def _parse_draft(content: str) -> ReActDraft:
    data = parse_json_object(content)
    candidates = [Candidate(**c) for c in data.get("candidates", []) if c.get("tmdb_id")]
    return ReActDraft(
        candidates=candidates,
        is_clarification=bool(data.get("is_clarification", False)),
        clarification_question=data.get("clarification_question"),
    )


async def _run_tool(
    tool_call, context: ReactContext, allowed: List[str]
) -> Tuple[str, dict, bool]:
    """Execute one tool call.

    Returns (result_text, result_obj, is_clarification).
    result_text is what we feed back to the LLM as the tool message content.
    """
    name = tool_call.function.name
    if not registry.exists(name) or name not in allowed:
        return (
            f"Error: tool '{name}' is not available. Available tools: {allowed}. "
            f"Try again with one of those.",
            {"error": "invalid_tool", "name": name},
            False,
        )
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return (
            f"Error: arguments for '{name}' were not valid JSON. Try again.",
            {"error": "invalid_arguments", "name": name},
            False,
        )

    if name == "tmdb_fallback_search":
        # Server-injected rather than trusted from the model: it's the same
        # value on every call this pass, and a mismatched/garbled copy would
        # silently produce wrong availability results.
        args["country"] = context.preferences.country
        args["platforms"] = context.preferences.platforms

    try:
        result = await registry.get_executor(name)(args)
    except RagUnavailable:
        # Let this propagate out of react_agent entirely — the orchestrator
        # retries the whole pass in fallback mode specifically for this
        # failure (see _run_react_with_fallback), rather than treating it as
        # a self-correctable tool error the model can retry its way out of.
        raise
    except Exception as exc:  # tool self-corrects on next cycle
        return (
            f"Error executing {name}: {exc}",
            {"error": "execution_failed", "name": name, "detail": str(exc)},
            False,
        )

    if name in ("rag_search", "tmdb_fallback_search"):
        result = _drop_known_movies(result, context)

    is_clarify = name == ask_user_clarification.TOOL_NAME and result.get("action") == "clarify"
    return json.dumps(result, ensure_ascii=False), result, is_clarify


def _drop_known_movies(result: dict, context: ReactContext) -> dict:
    """Filter out results already approved or excluded in an earlier pass.

    Without this, the model can waste a whole pass re-drafting (and the
    orchestrator re-verifying) a movie it already ruled out or already
    banked — the exclusion list only existed as prompt text, which the model
    doesn't always heed, especially several tool calls into a long pass.
    """
    known_ids = {c.tmdb_id for c in context.approved} | {
        e.tmdb_id for e in context.excluded
    }
    results = result.get("results") or []
    kept = [r for r in results if r.get("tmdb_id") not in known_ids]
    filtered_out = len(results) - len(kept)
    out = dict(result, results=kept, count=len(kept))
    if filtered_out:
        out["filtered_out"] = filtered_out
    return out


async def react_agent(context: ReactContext, steps: List[dict]) -> ReActDraft:
    rag_budget = _rag_budget_for_pass(context.pass_number)
    system_prompt = build_react_system_prompt(
        context.feedback,
        context.use_fallback,
        context.interactive,
        approved=[c.model_dump() for c in context.approved],
        excluded=[e.model_dump() for e in context.excluded],
        is_final_pass=context.is_final_pass,
        pass_number=context.pass_number,
        rag_budget=rag_budget,
    )

    user_context = _build_user_context(context)
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_context},
    ]

    consecutive_invalid = 0
    rag_calls_made = 0

    for _ in range(config.MAX_TOOL_CALLS):
        rag_remaining = rag_budget == -1 or rag_calls_made < rag_budget
        allowed = _allowed_tools(context, rag_remaining)
        tools = registry.schemas_for(allowed)
        resp = await llm_client.chat(messages, tools=tools)
        message = resp.choices[0].message

        # No tool calls -> the draft is ready.
        if not message.tool_calls:
            log_step(
                steps,
                module=MODULE,
                system_prompt=system_prompt,
                user_prompt=user_context if not steps else "[continue agentic loop]",
                response={"content": message.content, "tool_calls": []},
            )
            return _parse_draft(message.content or "{}")

        # Only log a turn that carries a thought worth showing — a bare
        # tool-dispatch with no content adds nothing the following
        # ReActAgent/<tool> steps don't already show.
        if message.content:
            log_step(
                steps,
                module=MODULE,
                system_prompt=system_prompt,
                user_prompt=user_context if not steps else "[continue agentic loop]",
                response={
                    "content": message.content,
                    "tool_calls": [tc.function.name for tc in message.tool_calls],
                },
            )

        messages.append(_assistant_message(message))

        # Tool calls within one assistant turn are independent — run them
        # concurrently instead of awaiting them one at a time.
        tool_results = await asyncio.gather(
            *(_run_tool(tc, context, allowed) for tc in message.tool_calls)
        )

        clarify_result: Optional[dict] = None
        for tool_call, (result_text, result_obj, is_clarify) in zip(
            message.tool_calls, tool_results
        ):
            log_step(
                steps,
                module=f"{MODULE}/{tool_call.function.name}",
                system_prompt=None,
                user_prompt=tool_call.function.arguments,
                response=result_obj,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

            if is_clarify and clarify_result is None:
                clarify_result = result_obj

            if tool_call.function.name == "rag_search" and not (
                isinstance(result_obj, dict) and "error" in result_obj
            ):
                rag_calls_made += 1

            # Track consecutive invalid tool calls and abort if too many (§12).
            if isinstance(result_obj, dict) and result_obj.get("error") in (
                "invalid_tool",
                "invalid_arguments",
            ):
                consecutive_invalid += 1
            else:
                consecutive_invalid = 0

        # Short-circuit: the agent asked the user a question.
        if clarify_result is not None:
            return ReActDraft(
                candidates=[],
                is_clarification=True,
                clarification_question=clarify_result.get("question"),
            )

        if consecutive_invalid >= config.MAX_CONSECUTIVE_INVALID_TOOL_CALLS:
            raise RuntimeError(
                "ReAct agent made too many invalid tool calls in a row."
            )

    # Exhausted the tool-call budget: force a final draft with no tools.
    resp = await llm_client.chat(messages + [
        {
            "role": "user",
            "content": (
                "Stop searching now and return your final candidate list as the "
                "specified JSON object."
            ),
        }
    ])
    final = resp.choices[0].message
    log_step(
        steps,
        module=MODULE,
        system_prompt=system_prompt,
        user_prompt="[force final draft — tool budget exhausted]",
        response={"content": final.content},
    )
    return _parse_draft(final.content or "{}")
