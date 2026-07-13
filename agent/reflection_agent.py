"""Stage 2 — Reflection Agent.

Critiques a draft using the verify_recommendation tool (grounded in live TMDB
data), then returns an approve/reject/clarify verdict. On approve it composes the
final user-facing response itself — there is no separate composer stage.
"""
from __future__ import annotations

import json
from typing import List

from agent.clients.llm_client import llm_client
from agent.models import ReActDraft, ReflectionVerdict, UserPreferences
from agent.parsing import parse_json_object
from agent.prompts import build_reflection_system_prompt
from agent.steps import log_step
from agent.tools import ask_user_clarification, registry

MODULE = "ReflectionAgent"

# Reflection is a short loop: verify (maybe twice) then a verdict.
_MAX_REFLECTION_CYCLES = 4


def _allowed_tools(interactive: bool) -> List[str]:
    """verify_recommendation always; clarification only when interactive."""
    tools = ["verify_recommendation"]
    if interactive:
        tools.append("ask_user_clarification")
    return tools


def _build_user_context(draft: ReActDraft, preferences: UserPreferences) -> str:
    return (
        "User preferences (JSON):\n"
        + json.dumps(preferences.model_dump(), ensure_ascii=False)
        + "\n\nDraft candidates to critique (JSON):\n"
        + json.dumps([c.model_dump() for c in draft.candidates], ensure_ascii=False)
        + "\n\nVerify these against live data, then return your verdict JSON."
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


def _parse_verdict(content: str) -> ReflectionVerdict:
    data = parse_json_object(content)
    return ReflectionVerdict(
        decision=data.get("decision", "reject"),
        final_response=data.get("final_response"),
        critique=data.get("critique"),
        use_fallback=bool(data.get("use_fallback", False)),
        question=data.get("question"),
    )


async def reflection_agent(
    draft: ReActDraft,
    preferences: UserPreferences,
    steps: List[dict],
    interactive: bool = False,
) -> ReflectionVerdict:
    allowed = _allowed_tools(interactive)
    tools = registry.schemas_for(allowed)
    system_prompt = build_reflection_system_prompt(interactive)
    user_context = _build_user_context(draft, preferences)
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_context},
    ]

    for cycle in range(_MAX_REFLECTION_CYCLES):
        resp = await llm_client.chat(messages, tools=tools)
        message = resp.choices[0].message

        log_step(
            steps,
            module=MODULE,
            system_prompt=system_prompt if cycle == 0 else None,
            user_prompt=user_context if cycle == 0 else "[continue critique]",
            response={
                "content": message.content,
                "tool_calls": [tc.function.name for tc in (message.tool_calls or [])],
            },
        )

        # No tool calls -> the verdict is ready.
        if not message.tool_calls:
            return _parse_verdict(message.content or "{}")

        messages.append(_assistant_message(message))

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            if not registry.exists(name) or name not in allowed:
                result_text = (
                    f"Error: tool '{name}' is not available here. Available: {allowed}."
                )
                result_obj = {"error": "invalid_tool", "name": name}
            else:
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    result_obj = await registry.get_executor(name)(args)
                    result_text = json.dumps(result_obj, ensure_ascii=False)
                except Exception as exc:
                    # TMDBUnavailable and other errors propagate for verify;
                    # re-raise so the orchestrator returns a proper error status.
                    from agent.tools.verify_recommendation import TMDBUnavailable

                    if isinstance(exc, TMDBUnavailable):
                        raise
                    result_obj = {"error": "execution_failed", "detail": str(exc)}
                    result_text = f"Error executing {name}: {exc}"

            log_step(
                steps,
                module=f"{MODULE}/{name}",
                system_prompt=None,
                user_prompt=tool_call.function.arguments,
                response=result_obj,
            )

            # ask_user_clarification short-circuits to a clarify verdict.
            if (
                name == ask_user_clarification.TOOL_NAME
                and isinstance(result_obj, dict)
                and result_obj.get("action") == "clarify"
            ):
                return ReflectionVerdict(
                    decision="clarify", question=result_obj.get("question")
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    # Exhausted cycles: force a verdict with no tools.
    resp = await llm_client.chat(
        messages
        + [{"role": "user", "content": "Return your final verdict JSON now."}]
    )
    final = resp.choices[0].message
    log_step(
        steps,
        module=MODULE,
        system_prompt=None,
        user_prompt="[force final verdict]",
        response={"content": final.content},
    )
    return _parse_verdict(final.content or "{}")
