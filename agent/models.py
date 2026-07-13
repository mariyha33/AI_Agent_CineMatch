"""Pydantic models for every data shape flowing through the pipeline."""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# --- Stage 0: parsed user taste ----------------------------------------------
class UserPreferences(BaseModel):
    """Structured taste parsed from a natural-language request (Stage 0)."""

    mood: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    similar_to: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    country: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    min_rating: Optional[float] = None


# --- Stage 1: ReAct draft ----------------------------------------------------
class Candidate(BaseModel):
    """A single movie candidate produced by the ReAct agent."""

    title: str
    year: Optional[int] = None
    tmdb_id: int
    genres: List[str] = Field(default_factory=list)
    overview: Optional[str] = None
    rationale: Optional[str] = None


class ReActDraft(BaseModel):
    """Output of the ReAct agent for one pass."""

    candidates: List[Candidate] = Field(default_factory=list)
    is_clarification: bool = False
    clarification_question: Optional[str] = None


class ReactContext(BaseModel):
    """State passed into each ReAct pass by the orchestrator."""

    preferences: UserPreferences
    feedback: Optional[str] = None
    use_fallback: bool = False
    # True only when the caller is interactive (a GUI that can relay a follow-up
    # question). Gates whether ask_user_clarification is offered to the agent.
    interactive: bool = False


# --- Stage 2: Reflection verdict ---------------------------------------------
class ReflectionVerdict(BaseModel):
    """Output of the Reflection agent."""

    decision: str  # "approve" | "reject" | "clarify"
    final_response: Optional[str] = None
    critique: Optional[str] = None
    use_fallback: bool = False
    question: Optional[str] = None


class VerifyResult(BaseModel):
    """Per-candidate result returned by the verify_recommendation tool."""

    tmdb_id: int
    title: str
    available: bool
    platform: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    overview: Optional[str] = None
    popularity: Optional[float] = None
    keyword_tags: List[str] = Field(default_factory=list)
    verdict: str  # "pass" | "fail"
    reason: str


# --- Trace / API shapes ------------------------------------------------------
class StepPrompt(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None


class Step(BaseModel):
    module: str
    prompt: StepPrompt
    response: Any


class Message(BaseModel):
    role: str
    content: str


class ExecuteRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Message]] = None
    session_id: Optional[str] = None


class ExecuteResponse(BaseModel):
    status: str  # "ok" | "error"
    error: Optional[str] = None
    response: Optional[str] = None
    steps: List[Any] = Field(default_factory=list)
