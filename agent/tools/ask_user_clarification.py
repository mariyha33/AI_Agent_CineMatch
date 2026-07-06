"""ask_user_clarification — signal that the agent needs more information.

No external API call. Returns a signal object the orchestrator catches to
short-circuit the pipeline and return the question to the user. The next user
message (carrying conversation_history) starts a fresh pipeline run.
"""
from __future__ import annotations

TOOL_NAME = "ask_user_clarification"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Ask the user a single clarifying question when the request is too "
            "ambiguous to proceed (e.g. missing country/platform, or a genuine "
            "taste fork). Stops the pipeline and returns the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user.",
                }
            },
            "required": ["question"],
        },
    },
}


async def execute(args: dict) -> dict:
    question = args.get("question", "").strip()
    return {"action": "clarify", "question": question}
