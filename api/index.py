"""CineMatch FastAPI app — all /api/* routes.

Deployed as a single Vercel Python serverless function (see vercel.json).
"""
from __future__ import annotations

import os
import sys

# Ensure the project root (which contains the `agent` package) is importable
# whether run locally (uvicorn api.index:app) or on Vercel.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

from agent.agent_info import AGENT_INFO  # noqa: E402
from agent.models import ExecuteRequest  # noqa: E402
from agent.orchestrator import run_pipeline  # noqa: E402
from agent.tools.verify_recommendation import TMDBUnavailable  # noqa: E402

app = FastAPI(title="CineMatch", version="1.0.0")

# Open access — no auth guards (a future frontend calls this from the browser).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ARCHITECTURE_PNG = os.path.join(_ROOT, "public", "architecture.png")


# --- A. Team info ------------------------------------------------------------
@app.get("/api/team_info")
async def team_info() -> dict:
    return {
        "group_batch_order_number": "PLACEHOLDER",
        "team_name": "CineMatch",
        "students": [
            {"name": "Student A", "email": "a@example.com"},
            {"name": "Student B", "email": "b@example.com"},
            {"name": "Student C", "email": "c@example.com"},
        ],
    }


# --- B. Agent info -----------------------------------------------------------
@app.get("/api/agent_info")
async def agent_info() -> dict:
    return AGENT_INFO


# --- C. Model architecture (PNG) ---------------------------------------------
@app.get("/api/model_architecture")
async def model_architecture():
    if not os.path.exists(_ARCHITECTURE_PNG):
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error": (
                    "Architecture diagram not found. Place the PNG at "
                    "public/architecture.png."
                ),
            },
        )
    return FileResponse(_ARCHITECTURE_PNG, media_type="image/png")


# --- D. Execute (main entry point) -------------------------------------------
@app.post("/api/execute")
async def execute(req: ExecuteRequest) -> dict:
    steps: list = []
    try:
        response_text = await run_pipeline(
            prompt=req.prompt,
            conversation_history=req.conversation_history,
            steps=steps,
        )
        result = {
            "status": "ok",
            "error": None,
            "response": response_text,
            "steps": steps,
        }
        _log_run(req, response_text, steps, "ok", None)
        return result

    except TMDBUnavailable as exc:
        _log_run(req, None, steps, "error", str(exc))
        return {
            "status": "error",
            "error": str(exc),
            "response": None,
            "steps": [],
        }
    except Exception as exc:  # any unhandled failure -> graceful error response
        message = f"CineMatch failed to process this request: {exc}"
        _log_run(req, None, steps, "error", message)
        return {
            "status": "error",
            "error": message,
            "response": None,
            "steps": [],
        }


def _log_run(
    req: ExecuteRequest,
    response_text,
    steps: list,
    status: str,
    error,
) -> None:
    """Best-effort persistence to Supabase — never breaks the request."""
    try:
        from agent.clients.supabase_client import supabase_client

        supabase_client.save_pipeline_run(
            prompt=req.prompt,
            response=response_text,
            steps=steps,
            status=status,
            error=error,
            session_id=req.session_id,
        )
    except Exception:
        pass


# Local dev entry point: `python -m api.index` or `uvicorn api.index:app`.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
