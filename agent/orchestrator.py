"""Deterministic orchestrator — plain Python control flow, no LLM routing.

Runs Stage 0 (taste extraction) then the bounded ReAct <-> Reflection loop.
Availability is verified here, directly and deterministically (no LLM round
-trip) right after each ReAct draft; only availability-confirmed candidates
reach Reflection, which then judges taste only. Approved and excluded movies
accumulate across passes so ReAct keeps searching for genuinely new titles
instead of re-finding ones already accepted or ruled out.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from agent import config
from agent.models import (
    Candidate,
    ExcludedMovie,
    Message,
    ReactContext,
    ReActDraft,
    SessionState,
    UserPreferences,
)
from agent.react_agent import react_agent
from agent.reflection_agent import reflection_agent
from agent.steps import log_step
from agent.taste_extractor import taste_extractor
from agent.tools import verify_recommendation
from agent.tools.rag_search import RagUnavailable

VERIFY_MODULE = "Orchestrator/verify_recommendation"


async def run_pipeline(
    prompt: str,
    conversation_history: Optional[List[Message]],
    steps: List[dict],
    prior_state: Optional[SessionState] = None,
) -> Tuple[str, SessionState]:
    # Interactive (GUI) callers send conversation_history — even an empty list on
    # turn 1 — so they can relay a follow-up question. Automated/eval callers omit
    # it (None); for them the pipeline must ALWAYS return recommendations, never a
    # question. This gates whether ask_user_clarification is offered downstream.
    interactive = conversation_history is not None

    # --- Stage 0 — Taste Extraction -----------------------------------------
    preferences = await taste_extractor(prompt, conversation_history, steps)
    prior_state = prior_state or SessionState()

    # A request for something CineMatch fundamentally can't serve (TV
    # episodes, etc.) must be disclosed, not silently answered with movies as
    # if that's what was asked. Interactively, ask before burning any passes;
    # non-interactively (must always return something), disclose up front and
    # proceed with the movie pipeline anyway.
    if preferences.out_of_scope and interactive:
        return (
            f"{preferences.out_of_scope} I can look for movies with a similar "
            "feel instead — want me to try that?",
            SessionState(excluded=prior_state.excluded, recommended=prior_state.recommended),
        )
    disclosure_prefix = (
        f"{preferences.out_of_scope} Showing movie recommendations instead, "
        "since that's what I can do:\n\n"
        if preferences.out_of_scope
        else ""
    )

    # A platform constraint can never be availability-checked without a
    # country (TMDB's discover/watch-providers endpoints both require a
    # resolved region) — asking ReAct to "figure it out" burns whole passes
    # producing candidates that verify_recommendation then fails for a
    # misleading reason (it looks like "not available" when the real problem
    # is "couldn't check"). Handle it deterministically up front.
    if preferences.platforms and not preferences.country:
        if interactive:
            return (
                "Which country should I check "
                f"{', '.join(preferences.platforms)} availability for?",
                SessionState(excluded=prior_state.excluded, recommended=prior_state.recommended),
            )
        # Non-interactive: must always return something, but there is no
        # point spending multiple ReAct/Reflection passes on an availability
        # check that can never succeed — run one pass and disclose plainly
        # that availability wasn't confirmed, rather than pretending it was.
        no_country_context = ReactContext(
            preferences=preferences,
            interactive=False,
            use_fallback=False,
            pass_number=1,
            is_final_pass=True,
            excluded=_seed_excluded(prior_state),
        )
        draft = await _run_react_with_fallback(no_country_context, steps)
        response_text = disclosure_prefix + _compose_best_effort(draft, preferences)
        return response_text, _build_session_state(
            no_country_context, prior_state, draft.candidates
        )

    response_text, state = await _run_search_loop(
        preferences, prior_state, interactive, steps
    )
    return disclosure_prefix + response_text, state


async def _run_search_loop(
    preferences: UserPreferences,
    prior_state: SessionState,
    interactive: bool,
    steps: List[dict],
) -> Tuple[str, SessionState]:
    """The bounded ReAct <-> Reflection loop, run once country/scope checks
    above have already passed. Country is guaranteed resolvable here whenever
    platforms are set, and the request is known to be in-scope."""
    context = ReactContext(
        preferences=preferences,
        feedback=None,
        use_fallback=False,
        interactive=interactive,
        approved=[],
        # Cross-turn memory: movies already ruled out, and movies already shown
        # to the user in a previous turn, both feed in as hard excludes so this
        # turn's ReAct pass searches for genuinely new titles instead of
        # re-running the whole process against movies it already explored.
        excluded=_seed_excluded(prior_state),
        remaining_needed=config.MAX_CANDIDATES,
        is_final_pass=False,
    )

    draft: ReActDraft = ReActDraft()
    for _pass in range(1, config.MAX_PASSES + 1):
        context.pass_number = _pass
        context.is_final_pass = _pass == config.MAX_PASSES
        context.remaining_needed = max(
            config.MAX_CANDIDATES - len(context.approved), 1
        )
        # The RAG index has no availability signal (mediocre hit rate — see
        # Agentic loop overhaul below), so from pass 2 on tmdb_fallback_search
        # (availability-filtered by TMDB itself) becomes available regardless
        # of what triggered fallback mode so far.
        if _pass >= 2:
            context.use_fallback = True

        draft = await _run_react_with_fallback(context, steps)

        # ReAct asked the user a question -> short-circuit (interactive only,
        # and never on the final pass — there's no more retrying to clarify
        # into). In non-interactive mode a stray clarification is ignored.
        if draft.is_clarification and interactive and not context.is_final_pass:
            question = draft.clarification_question or "Could you clarify your request?"
            return question, _build_session_state(context, prior_state, [])

        known_ids = {c.tmdb_id for c in context.approved} | {
            e.tmdb_id for e in context.excluded
        }
        new_candidates = [c for c in draft.candidates if c.tmdb_id not in known_ids]

        if not new_candidates:
            if context.is_final_pass:
                break
            context.feedback = (
                "Your last draft contained no candidates beyond what's already "
                "approved or excluded. Search for genuinely different movies."
            )
            continue

        # --- Deterministic availability check (no LLM call) -----------------
        verify_result = await _verify_availability(new_candidates, preferences, steps)

        if verify_result.get("error") in ("missing_country", "unknown_country"):
            # Availability can never be confirmed for this country/platform
            # combo — no amount of retrying the ReAct/Reflection loop fixes
            # that, so stop burning passes on it (defense in depth: the
            # missing-country case is normally caught before this loop even
            # starts, but the country the taste extractor produced may still
            # fail to resolve to a TMDB region).
            if interactive and not context.is_final_pass:
                question = (
                    verify_result.get("message")
                    or "I need a valid country to check availability."
                ) + " What country should I use?"
                return question, _build_session_state(context, prior_state, [])
            shown = [
                c
                for c in new_candidates
                if c.tmdb_id not in {e.tmdb_id for e in context.excluded}
            ]
            response_text = _compose_best_effort(ReActDraft(candidates=shown), preferences)
            return response_text, _build_session_state(context, prior_state, shown)

        verified = verify_result.get("results", [])

        available: List[dict] = []
        for v in verified:
            if v.get("verdict") == "pass":
                available.append(v)
            else:
                context.excluded.append(
                    ExcludedMovie(
                        tmdb_id=v["tmdb_id"],
                        title=v.get("title") or "",
                        reason=v.get("reason") or "Not available.",
                    )
                )

        if not available:
            if context.is_final_pass:
                break
            context.feedback = (
                "None of your candidates were available on the requested "
                "platform/country. Search for different, verifiably available "
                "movies."
            )
            context.use_fallback = True
            continue

        # --- Taste judgment (single LLM call) --------------------------------
        verdict = await reflection_agent(
            available,
            preferences,
            context.approved,
            steps,
            interactive,
            context.is_final_pass,
        )

        if verdict.decision == "clarify":
            if interactive and not context.is_final_pass:
                question = verdict.question or "Could you clarify your request?"
                return question, _build_session_state(context, prior_state, [])
            context.feedback = (
                "Do not ask the user questions. Make reasonable assumptions about "
                "any missing details and return at least one recommendation."
            )
            context.use_fallback = verdict.use_fallback
            continue

        _apply_verdict(verdict, available, new_candidates, context)

        # Defense in depth: don't let a premature "approve" (below
        # MIN_CANDIDATES, not the final pass) end the pipeline early just
        # because the model treated "bank this one hard-constraint match" as
        # license to stop searching. Reflection's prompt tells it to keep
        # "decision" at "reject" until enough candidates have accumulated,
        # but this is a real trace-observed failure mode worth guarding in
        # code rather than trusting the model's count-math alone.
        premature_approve = (
            verdict.decision == "approve"
            and not context.is_final_pass
            and len(context.approved) < config.MIN_CANDIDATES
        )

        if verdict.decision == "approve" and not premature_approve:
            response_text = verdict.final_response or _compose_final(context.approved, preferences)
            return response_text, _build_session_state(context, prior_state, context.approved)

        # Enough movies have accumulated approval across passes even though
        # this pass's overall decision was "reject" (e.g. a mixed batch) ->
        # stop early instead of spending another ReAct/Reflection round.
        if len(context.approved) >= config.MAX_CANDIDATES:
            return _compose_final(context.approved, preferences), _build_session_state(
                context, prior_state, context.approved
            )

        if context.is_final_pass:
            break

        if premature_approve:
            # The model thought it was done, so it likely left "critique"
            # empty — synthesize feedback rather than running the next pass
            # with no guidance at all.
            context.feedback = verdict.critique or (
                f"You have only {len(context.approved)} approved movie(s) so "
                f"far, below the required {config.MIN_CANDIDATES}. Keep "
                "searching for more genuinely new matching movies before "
                "this can be approved."
            )
        else:
            context.feedback = verdict.critique
        context.use_fallback = verdict.use_fallback

    # Passes exhausted -> best effort from whatever accumulated.
    if context.approved:
        return _compose_final(context.approved, preferences), _build_session_state(
            context, prior_state, context.approved
        )
    # Never present a candidate here that verify_recommendation already
    # confirmed unavailable or Reflection already rejected — draft.candidates
    # is just the LAST pass's raw draft, which may be exactly the batch that
    # just failed verification (see the "None of your candidates were
    # available" feedback path above).
    excluded_ids = {e.tmdb_id for e in context.excluded}
    shown = [c for c in draft.candidates if c.tmdb_id not in excluded_ids]
    response_text = _compose_best_effort(ReActDraft(candidates=shown), preferences)
    return response_text, _build_session_state(context, prior_state, shown)


def _seed_excluded(prior_state: SessionState) -> List[ExcludedMovie]:
    """Carry prior-turn exclusions and prior-turn recommendations forward as
    hard excludes, so this turn's ReAct pass never re-drafts a movie it
    already ruled out or already showed the user in an earlier turn."""
    seeded: List[ExcludedMovie] = list(prior_state.excluded)
    seen_ids = {e.tmdb_id for e in seeded}
    for r in prior_state.recommended:
        if r.tmdb_id in seen_ids:
            continue
        seeded.append(
            ExcludedMovie(
                tmdb_id=r.tmdb_id,
                title=r.title,
                reason="Already recommended in a previous turn — do not repeat.",
            )
        )
        seen_ids.add(r.tmdb_id)
    return seeded


def _build_session_state(
    context: ReactContext,
    prior_state: SessionState,
    recommended_this_turn: List[Candidate],
) -> SessionState:
    """Fold this turn's outcome into the state the client round-trips."""
    excluded_by_id = {e.tmdb_id: e for e in context.excluded}

    recommended_by_id = {r.tmdb_id: r for r in prior_state.recommended}
    for c in recommended_this_turn:
        recommended_by_id[c.tmdb_id] = ExcludedMovie(
            tmdb_id=c.tmdb_id,
            title=c.title,
            reason="Recommended in a previous turn.",
        )

    return SessionState(
        excluded=list(excluded_by_id.values()),
        recommended=list(recommended_by_id.values()),
    )


async def _run_react_with_fallback(
    context: ReactContext, steps: List[dict]
) -> ReActDraft:
    """Run ReAct; if the RAG backend itself is unreachable, retry once forcing
    fallback. Only RagUnavailable is caught here — anything else (a JSON
    parse crash, an LLM auth/connection error, the "too many invalid tool
    calls" RuntimeError) is a real bug, not a recoverable RAG outage, and
    should surface immediately instead of silently doubling the cost of this
    pass while masking the actual problem."""
    try:
        return await react_agent(context, steps)
    except RagUnavailable:
        if context.use_fallback:
            raise  # already in fallback mode; surface the error
        context.use_fallback = True
        return await react_agent(context, steps)


async def _verify_availability(
    candidates: List[Candidate], preferences: UserPreferences, steps: List[dict]
) -> dict:
    """Run verify_recommendation directly — deterministic, parallel, no LLM.

    Returns the raw tool result dict (not just the "results" list) so the
    caller can detect a batch-level "error" (missing/unresolvable country) and
    stop retrying instead of treating every candidate as individually
    unavailable for a misleading reason.
    """
    args = {
        "candidates": [
            {"tmdb_id": c.tmdb_id, "title": c.title} for c in candidates
        ],
        "country": preferences.country,
        "platforms": preferences.platforms,
        "user_mood": preferences.mood,
        "exclude_people": preferences.exclude_people,
    }
    result = await verify_recommendation.execute(args)
    log_step(
        steps,
        module=VERIFY_MODULE,
        system_prompt=None,
        user_prompt=None,
        response=result,
    )
    return result


def _apply_verdict(
    verdict, available: List[dict], new_candidates: List[Candidate], context: ReactContext
) -> None:
    """Fold this pass's per-candidate outcomes into cross-pass approved/excluded state.

    Every candidate the Reflection agent was given this pass must land in
    exactly one bucket. If the model's response leaves one in neither list (or
    puts it in both), treat it as rejected rather than letting it silently
    vanish — otherwise ReAct could re-surface it next pass since it's absent
    from both context.approved and context.excluded (see _drop_known_movies).
    """
    approved_ids = set(verdict.approved_ids)
    rejected_ids = {r.tmdb_id for r in verdict.rejected}
    conflicting_ids = approved_ids & rejected_ids
    if conflicting_ids:
        approved_ids -= conflicting_ids

    verified_by_id = {v["tmdb_id"]: v for v in available}
    original_by_id = {c.tmdb_id: c for c in new_candidates}

    for aid in approved_ids:
        v = verified_by_id.get(aid)
        if v is None:
            continue
        orig = original_by_id.get(aid)
        context.approved.append(
            Candidate(
                title=v.get("title") or (orig.title if orig else "Unknown"),
                year=orig.year if orig else None,
                tmdb_id=aid,
                genres=v.get("genres") or (orig.genres if orig else []),
                overview=v.get("overview") or (orig.overview if orig else None),
                rationale=orig.rationale if orig else None,
                platform=v.get("platform"),
            )
        )

    for r in verdict.rejected:
        context.excluded.append(
            ExcludedMovie(tmdb_id=r.tmdb_id, title=r.title, reason=r.reason)
        )

    unaccounted_ids = set(verified_by_id.keys()) - approved_ids - rejected_ids
    for uid in unaccounted_ids:
        v = verified_by_id[uid]
        context.excluded.append(
            ExcludedMovie(
                tmdb_id=uid,
                title=v.get("title") or "Unknown",
                reason="Reflection did not explicitly approve or reject this candidate.",
            )
        )


def _compose_final(approved: List[Candidate], preferences: UserPreferences) -> str:
    """Format the accumulated, availability-confirmed, taste-approved list."""
    country = preferences.country or "your region"
    lines: List[str] = []
    for c in approved[: config.MAX_CANDIDATES]:
        year = f" ({c.year})" if c.year else ""
        rationale = c.rationale or "Matches the taste you described."
        platform = c.platform or (
            preferences.platforms[0] if preferences.platforms else "a supported service"
        )
        lines.append(f"**{c.title}{year}** — {rationale}")
        lines.append(f"↳ Available on {platform} — {country}")
        lines.append("")
    return "\n".join(lines).strip()


def _compose_best_effort(draft: ReActDraft, preferences: UserPreferences) -> str:
    """Format whatever candidates we have when nothing was ever approved."""
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
