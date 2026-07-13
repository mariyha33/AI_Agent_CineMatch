"""Stage 1 — ReAct Agent.

A true agentic loop using the LLM's native function-calling. The LLM decides
which tool to call, in what order, and when to stop. When it stops (returns text
with no tool calls) it must emit a JSON ReActDraft.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from agent import config
from agent.clients.llm_client import llm_client
from agent.models import Candidate, ReactContext, ReActDraft
from agent.parsing import parse_json_object
from agent.prompts import build_react_system_prompt
from agent.steps import log_step
from agent.tools import ask_user_clarification, registry

MODULE = "ReActAgent"


def _allowed_tools(context: ReactContext) -> List[str]:
    """rag_search always; fallback when enabled; clarification only interactively."""
    tools = ["rag_search"]
    if context.use_fallback:
        tools.append("tmdb_fallback_search")
    if context.interactive:
        tools.append("ask_user_clarification")
    return tools


def _build_user_context(context: ReactContext) -> str:
    prefs = context.preferences
    return (
        "User preferences (JSON):\n"
        + json.dumps(prefs.model_dump(), ensure_ascii=False)
        + "\n\nFind 1-5 matching movies. Aim for "
        + f"at least {config.MIN_CANDIDATES} candidates when possible."
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

    try:
        result = await registry.get_executor(name)(args)
    except Exception as exc:  # tool self-corrects on next cycle
        return (
            f"Error executing {name}: {exc}",
            {"error": "execution_failed", "name": name, "detail": str(exc)},
            False,
        )

    is_clarify = name == ask_user_clarification.TOOL_NAME and result.get("action") == "clarify"
    return json.dumps(result, ensure_ascii=False), result, is_clarify


async def react_agent(context: ReactContext, steps: List[dict]) -> ReActDraft:
    system_prompt = build_react_system_prompt(
        context.feedback, context.use_fallback, context.interactive
    )
    allowed = _allowed_tools(context)
    tools = registry.schemas_for(allowed)

    user_context = _build_user_context(context)
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_context},
    ]

    consecutive_invalid = 0

    for _ in range(config.MAX_TOOL_CALLS):
        resp = await llm_client.chat(messages, tools=tools)
        message = resp.choices[0].message

        log_step(
            steps,
            module=MODULE,
            system_prompt=system_prompt,
            user_prompt=user_context if not steps else "[continue agentic loop]",
            response={
                "content": message.content,
                "tool_calls": [tc.function.name for tc in (message.tool_calls or [])],
            },
        )

        # No tool calls -> the draft is ready.
        if not message.tool_calls:
            return _parse_draft(message.content or "{}")

        messages.append(_assistant_message(message))

        for tool_call in message.tool_calls:
            result_text, result_obj, is_clarify = await _run_tool(
                tool_call, context, allowed
            )

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

            # Short-circuit: the agent asked the user a question.
            if is_clarify:
                return ReActDraft(
                    candidates=[],
                    is_clarification=True,
                    clarification_question=result_obj.get("question"),
                )

            # Track consecutive invalid tool calls and abort if too many (§12).
            if isinstance(result_obj, dict) and result_obj.get("error") in (
                "invalid_tool",
                "invalid_arguments",
            ):
                consecutive_invalid += 1
                if consecutive_invalid >= config.MAX_CONSECUTIVE_INVALID_TOOL_CALLS:
                    raise RuntimeError(
                        "ReAct agent made too many invalid tool calls in a row."
                    )
            else:
                consecutive_invalid = 0

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
