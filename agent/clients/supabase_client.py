"""Supabase read/write helper for run logging and conversation history.

All writes are best-effort — a logging failure must never break the pipeline
(the caller wraps these in try/except).
"""
from __future__ import annotations

from typing import Any, List, Optional

from supabase import create_client

from agent import config


class SupabaseClient:
    def __init__(self) -> None:
        self._client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    def save_pipeline_run(
        self,
        prompt: str,
        response: Optional[str],
        steps: List[dict],
        status: str,
        error: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Insert one row into pipeline_runs."""
        self._client.table("pipeline_runs").insert(
            {
                "session_id": session_id,
                "prompt": prompt,
                "response": response,
                "steps": steps,
                "status": status,
                "error": error,
            }
        ).execute()

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Insert one conversation turn into conversations."""
        self._client.table("conversations").insert(
            {"session_id": session_id, "role": role, "content": content}
        ).execute()

    def get_conversation(self, session_id: str) -> List[dict]:
        """Retrieve all turns for a session, oldest first."""
        resp = (
            self._client.table("conversations")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return resp.data or []


supabase_client = SupabaseClient()
