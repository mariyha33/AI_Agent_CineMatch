"""Static metadata served by GET /api/agent_info.

Kept separate from api/index.py because the example step traces are verbose.
Module names in the example `steps` traces match the real pipeline exactly.
"""
from __future__ import annotations

from agent.prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    REFLECTION_AGENT_SYSTEM_PROMPT,
    TASTE_EXTRACTOR_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

DESCRIPTION = (
    "CineMatch is a movie recommendation agent that personalizes picks to your "
    "taste and verifies each one is actually streamable/rentable on your "
    "platform in your country.\n\n"
    "What it CAN do: understand a natural-language or structured taste brief "
    "(mood, genres, movies you love, movies to exclude, country, platforms, "
    "year range, minimum rating); retrieve taste-matched candidates from an IMDB "
    "vector index (RAG); fall back to structured TMDB discovery for obscure or "
    "availability-constrained requests; verify real-time availability against "
    "TMDB before recommending; and ask a clarifying question when your request "
    "is ambiguous.\n\n"
    "What it CANNOT do (constraints): it does not recommend TV shows (movies "
    "only); it never asserts availability without a live TMDB lookup; it has no "
    "user accounts, watch history, or personal data beyond what you type; and it "
    "cannot honor runtime/duration filters from the RAG index alone (those are "
    "handled via TMDB discovery)."
)

PURPOSE = (
    "Solve the two things general LLMs get wrong for movie recommendations: "
    "genuine personalization (not just the most popular title) and real, "
    "country- and platform-specific streaming availability."
)

PROMPT_EXAMPLES = [
    {
        "prompt": (
            "I want something like Sicario and No Country for Old Men — slow-burn, "
            "tense, morally grey crime. I'm in Israel and have Netflix. I've "
            "already seen Prisoners."
        ),
        "full_response": (
            "**Wind River (2017)** — A slow-burn crime mystery with the same "
            "morally grey dread as Sicario (same writer, Taylor Sheridan), set "
            "against a bleak, snowbound landscape.\n"
            "↳ Available on Netflix — Israel\n\n"
            "**Hell or High Water (2016)** — Tense, character-driven modern-western "
            "crime with the quiet menace you liked in No Country for Old Men.\n"
            "↳ Available on Netflix — Israel"
        ),
        "steps": [
            {
                "module": "TasteExtractor",
                "prompt": {
                    "system_prompt": TASTE_EXTRACTOR_SYSTEM_PROMPT,
                    "user_prompt": (
                        "I want something like Sicario and No Country for Old Men — "
                        "slow-burn, tense, morally grey crime. I'm in Israel and "
                        "have Netflix. I've already seen Prisoners."
                    ),
                },
                "response": {
                    "mood": "slow-burn, tense, morally grey",
                    "genres": ["Crime", "Thriller"],
                    "similar_to": ["Sicario", "No Country for Old Men"],
                    "exclude": ["Prisoners"],
                    "country": "Israel",
                    "platforms": ["Netflix"],
                    "year_min": None,
                    "year_max": None,
                    "min_rating": None,
                },
            },
            {
                "module": "ReActAgent",
                "prompt": {
                    "system_prompt": REACT_AGENT_SYSTEM_PROMPT,
                    "user_prompt": (
                        "User preferences (JSON): {\"mood\": \"slow-burn, tense, "
                        "morally grey\", \"genres\": [\"Crime\", \"Thriller\"], "
                        "\"similar_to\": [\"Sicario\", \"No Country for Old Men\"], "
                        "\"exclude\": [\"Prisoners\"], \"country\": \"Israel\", "
                        "\"platforms\": [\"Netflix\"]}"
                    ),
                },
                "response": {
                    "content": None,
                    "tool_calls": ["rag_search"],
                },
            },
            {
                "module": "ReActAgent/rag_search",
                "prompt": {
                    "system_prompt": None,
                    "user_prompt": (
                        "{\"query_text\": \"slow-burn tense morally grey crime "
                        "thriller like Sicario and No Country for Old Men\", "
                        "\"genres\": [\"Crime\", \"Thriller\"], \"top_k\": 10}"
                    ),
                },
                "response": {
                    "results": [
                        {
                            "title": "Wind River",
                            "year": 2017,
                            "tmdb_id": 395834,
                            "genres": ["Crime", "Mystery", "Thriller"],
                            "score": 7.4,
                            "overview": "A veteran tracker helps an FBI agent "
                            "investigate a murder on a remote reservation.",
                            "pinecone_score": 0.89,
                        },
                        {
                            "title": "Hell or High Water",
                            "year": 2016,
                            "tmdb_id": 331482,
                            "genres": ["Crime", "Drama", "Thriller"],
                            "score": 7.6,
                            "overview": "Two brothers rob banks to save the "
                            "family ranch as a ranger closes in.",
                            "pinecone_score": 0.87,
                        },
                    ],
                    "count": 2,
                },
            },
            {
                "module": "ReActAgent",
                "prompt": {
                    "system_prompt": REACT_AGENT_SYSTEM_PROMPT,
                    "user_prompt": "[continue agentic loop]",
                },
                "response": {
                    "content": (
                        "{\"candidates\": [{\"title\": \"Wind River\", \"year\": "
                        "2017, \"tmdb_id\": 395834, \"genres\": [\"Crime\", "
                        "\"Thriller\"], \"overview\": \"...\", \"rationale\": "
                        "\"Same writer as Sicario; slow-burn, morally grey.\"}, "
                        "{\"title\": \"Hell or High Water\", \"year\": 2016, "
                        "\"tmdb_id\": 331482, \"genres\": [\"Crime\", \"Thriller\"], "
                        "\"overview\": \"...\", \"rationale\": \"Tense modern "
                        "western with quiet menace.\"}], \"is_clarification\": "
                        "false, \"clarification_question\": null}"
                    ),
                    "tool_calls": [],
                },
            },
            {
                "module": "ReflectionAgent",
                "prompt": {
                    "system_prompt": REFLECTION_AGENT_SYSTEM_PROMPT,
                    "user_prompt": (
                        "Draft candidates: Wind River (395834), Hell or High "
                        "Water (331482). Verify against live data, then return "
                        "your verdict JSON."
                    ),
                },
                "response": {
                    "content": None,
                    "tool_calls": ["verify_recommendation"],
                },
            },
            {
                "module": "ReflectionAgent/verify_recommendation",
                "prompt": {
                    "system_prompt": None,
                    "user_prompt": (
                        "{\"candidates\": [{\"tmdb_id\": 395834, \"title\": \"Wind "
                        "River\"}, {\"tmdb_id\": 331482, \"title\": \"Hell or High "
                        "Water\"}], \"country\": \"Israel\", \"platforms\": "
                        "[\"Netflix\"], \"user_mood\": \"slow-burn, morally grey\"}"
                    ),
                },
                "response": {
                    "results": [
                        {
                            "tmdb_id": 395834,
                            "title": "Wind River",
                            "available": True,
                            "platform": "Netflix",
                            "genres": ["Crime", "Mystery", "Thriller"],
                            "popularity": 21.3,
                            "keyword_tags": ["snow", "murder investigation", "revenge"],
                            "verdict": "pass",
                            "reason": "Available on Netflix in IL.",
                        },
                        {
                            "tmdb_id": 331482,
                            "title": "Hell or High Water",
                            "available": True,
                            "platform": "Netflix",
                            "genres": ["Crime", "Drama", "Thriller"],
                            "popularity": 18.7,
                            "keyword_tags": ["bank robbery", "texas", "brothers"],
                            "verdict": "pass",
                            "reason": "Available on Netflix in IL.",
                        },
                    ],
                    "count": 2,
                    "region": "IL",
                },
            },
            {
                "module": "ReflectionAgent",
                "prompt": {
                    "system_prompt": None,
                    "user_prompt": "[continue critique]",
                },
                "response": {
                    "decision": "approve",
                    "final_response": (
                        "**Wind River (2017)** — A slow-burn crime mystery with the "
                        "same morally grey dread as Sicario (same writer, Taylor "
                        "Sheridan), set against a bleak, snowbound landscape.\n"
                        "↳ Available on Netflix — Israel\n\n"
                        "**Hell or High Water (2016)** — Tense, character-driven "
                        "modern-western crime with the quiet menace you liked in No "
                        "Country for Old Men.\n↳ Available on Netflix — Israel"
                    ),
                    "critique": None,
                    "use_fallback": False,
                    "question": None,
                },
            },
        ],
    },
    {
        "prompt": "Recommend me a good movie to watch tonight.",
        "full_response": (
            "Happy to help! To make sure I recommend something you can actually "
            "stream and that fits your taste: which country are you in, which "
            "streaming platforms do you have (e.g. Netflix, Disney+), and what "
            "kind of mood are you after tonight (e.g. light and funny, tense "
            "thriller, feel-good)?"
        ),
        "steps": [
            {
                "module": "TasteExtractor",
                "prompt": {
                    "system_prompt": TASTE_EXTRACTOR_SYSTEM_PROMPT,
                    "user_prompt": "Recommend me a good movie to watch tonight.",
                },
                "response": {
                    "mood": None,
                    "genres": [],
                    "similar_to": [],
                    "exclude": [],
                    "country": None,
                    "platforms": [],
                    "year_min": None,
                    "year_max": None,
                    "min_rating": None,
                },
            },
            {
                "module": "ReActAgent",
                "prompt": {
                    "system_prompt": REACT_AGENT_SYSTEM_PROMPT,
                    "user_prompt": (
                        "User preferences (JSON): {\"mood\": null, \"genres\": [], "
                        "\"country\": null, \"platforms\": []}. Find 1-5 matching "
                        "movies."
                    ),
                },
                "response": {
                    "content": None,
                    "tool_calls": ["ask_user_clarification"],
                },
            },
            {
                "module": "ReActAgent/ask_user_clarification",
                "prompt": {
                    "system_prompt": None,
                    "user_prompt": (
                        "{\"question\": \"Which country are you in, which streaming "
                        "platforms do you have, and what mood are you after "
                        "tonight?\"}"
                    ),
                },
                "response": {
                    "action": "clarify",
                    "question": (
                        "Which country are you in, which streaming platforms do "
                        "you have, and what mood are you after tonight?"
                    ),
                },
            },
        ],
    },
]

AGENT_INFO = {
    "description": DESCRIPTION,
    "purpose": PURPOSE,
    "prompt_template": USER_PROMPT_TEMPLATE,
    "prompt_examples": PROMPT_EXAMPLES,
}
