"""Generate architecture_image/cinematch_architecture.png — CineMatch agent architecture diagram.

Module names match exactly what log_step records:
  TasteExtractor
  ReActAgent
  ReActAgent/rag_search
  ReActAgent/tmdb_fallback_search
  ReActAgent/ask_user_clarification
  Orchestrator/verify_recommendation
  ReflectionAgent
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "architecture_image", "cinematch_architecture.png")

# ── colours ────────────────────────────────────────────────────────────────────
C_ENTRY = "#1f3a5f"   # dark navy  – API / entry points
C_STAGE = "#1a7a6e"   # dark teal  – pipeline stages
C_TOOL  = "#2e8b57"   # forest-green – ReActAgent sub-tools
C_EXT   = "#555555"   # grey       – external services
C_OUT   = "#2d6e2d"   # dark green – final output
C_LOOP  = "#c8701e"   # amber      – reject / feedback loop
BG      = "#f4f6f9"

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor(BG)

# ── helpers ────────────────────────────────────────────────────────────────────
def box(x, y, w, h, label, sub="", color=C_STAGE, fontsize=9.5):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.09",
        linewidth=1.2, edgecolor="white",
        facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    ty = y + (0.14 if sub else 0)
    ax.text(x, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(x, y - 0.25, sub, ha="center", va="center",
                fontsize=7.2, color="#cde8e3", zorder=4)
    return x, y


def arrow(x0, y0, x1, y1, color="#888888", lw=1.4, style="-|>", rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=style, color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                ), zorder=2)


def alabel(x, y, text, fs=7.2):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="#444", fontstyle="italic",
            bbox=dict(fc=BG, ec="none", pad=1), zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN POSITIONS
#   left ext:  x=1.5  (Pinecone)
#   main pipe: x=6.4
#   right ext: x=13.6 (LLMOD, TMDB, Supabase)
# ══════════════════════════════════════════════════════════════════════════════

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(6.4, 10.6, "CineMatch — Agent Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1a2540")

# ── Row 1: API entry & Supabase ────────────────────────────────────────────────
box(3.5, 9.8, 2.8, 0.75, "GUI", "GET /  (public/index.html)", C_ENTRY, 9)
box(6.8, 9.8, 3.5, 0.75, "POST /api/execute", "api/index.py", C_ENTRY, 9)
box(13.6, 9.8, 2.4, 0.75, "Supabase", "best-effort run logging", C_EXT, 8.5)

arrow(4.9, 9.8, 5.05, 9.8)                       # GUI → execute
arrow(6.8, 9.43, 6.8, 8.95)                      # execute ↓
arrow(8.55, 9.8, 12.4, 9.8, color="#aaa", lw=1)  # execute → Supabase (logging)

# Orchestrator label (not a full box — it's implicit)
ax.text(6.8, 8.7, "Orchestrator  (agent/orchestrator.py)",
        ha="center", va="center", fontsize=7.8, color="#555", fontstyle="italic")

# ── Row 2: TasteExtractor ──────────────────────────────────────────────────────
box(6.8, 8.0, 3.5, 0.75, "TasteExtractor", "Stage 0 — taste extraction", C_STAGE)
arrow(6.8, 8.45, 6.8, 8.38)

# ── Row 3: ReActAgent ──────────────────────────────────────────────────────────
box(6.8, 6.9, 3.5, 0.75, "ReActAgent", "Stage 1 — bounded tool-calling loop", C_STAGE)
arrow(6.8, 7.63, 6.8, 7.28)

# ── Row 4: ReActAgent sub-tools ────────────────────────────────────────────────
box(2.2, 5.5, 3.0, 0.72, "ReActAgent/rag_search",             "", C_TOOL, 8.8)
box(6.8, 5.5, 3.2, 0.72, "ReActAgent/tmdb_fallback_search",   "", C_TOOL, 8.2)
box(11.5, 5.5, 3.2, 0.72, "ReActAgent/ask_user_clarification", "", C_TOOL, 8.0)

# ReActAgent → tools
arrow(6.8, 6.53, 2.2,  5.86, rad=-0.15)
arrow(6.8, 6.53, 6.8,  5.86, rad=0.0)
arrow(6.8, 6.53, 11.5, 5.86, rad=0.15)

# tools → ReActAgent (results back)
arrow(2.2,  5.14, 6.5,  6.53, rad=0.25, color="#aaa", lw=1.0)
arrow(6.8,  5.14, 6.8,  6.53, rad=0.0,  color="#aaa", lw=1.0)
arrow(11.5, 5.14, 7.1,  6.53, rad=-0.25, color="#aaa", lw=1.0)

# ── Row 5: Orchestrator/verify_recommendation ──────────────────────────────────
box(6.8, 4.1, 4.0, 0.78,
    "Orchestrator/verify_recommendation",
    "deterministic availability check — no LLM", C_STAGE, 8.8)

# ReActAgent draft → verify  (flows straight down past tools row)
arrow(6.8, 6.53, 6.8, 4.49, color="#777", lw=1.2, rad=0.0)
alabel(7.5, 5.5, "draft")

# ── Row 6: ReflectionAgent ─────────────────────────────────────────────────────
box(6.8, 2.85, 3.5, 0.75, "ReflectionAgent", "Stage 2 — taste / novelty judgment", C_STAGE)
arrow(6.8, 3.71, 6.8, 3.23)

# ── Row 7: Final Response ──────────────────────────────────────────────────────
box(6.8, 1.75, 2.8, 0.7, "Final Response", "", C_OUT, 10.5)
arrow(6.8, 2.48, 6.8, 2.1)

# ── Right column: LLMOD, TMDB ──────────────────────────────────────────────────
box(13.6, 8.0,  2.2, 0.7, "LLMOD",   "LLM client",               C_EXT, 8.5)
box(13.6, 5.5,  2.2, 0.7, "TMDB",    "movies / watch providers",  C_EXT, 8.5)

# Left column: Pinecone
box(1.1, 8.0, 1.9, 0.7, "Pinecone", "vector index", C_EXT, 8.5)

# TasteExtractor → LLMOD  (chat + embed)
arrow(8.55, 8.0, 12.5, 8.0, color="#999", lw=1.1)
alabel(10.7, 8.2, "chat (embed query → response)")

# ReflectionAgent → LLMOD  (one straight run up the right side)
arrow(8.55, 2.85, 12.5, 7.65, color="#999", lw=1.1, rad=-0.25)
alabel(11.8, 5.4, "chat", fs=7)

# ReActAgent/rag_search → Pinecone
arrow(2.2, 5.14, 1.8, 8.35, color="#999", lw=1.0, rad=0.3)
alabel(1.4, 6.9, "semantic\nquery", fs=7)

# ReActAgent/rag_search embed call → LLMOD  (via rag_search)
arrow(3.7, 5.5, 12.5, 7.75, color="#999", lw=1.0, rad=-0.1)
alabel(8.5, 7.0, "embed query", fs=7)

# ReActAgent/tmdb_fallback_search → TMDB
arrow(8.4, 5.5, 12.5, 5.5, color="#999", lw=1.0)
alabel(10.7, 5.72, "discover/movie")

# Orchestrator/verify_recommendation → TMDB
arrow(8.8, 4.1, 12.5, 5.15, color="#999", lw=1.0, rad=-0.15)
alabel(11.2, 4.5, "detail + watch\nproviders", fs=7)

# ── Reject / feedback loop (amber) ─────────────────────────────────────────────
# runs: ReflectionAgent → left → up → ReActAgent left edge
lx = 4.6   # x of the vertical part
ax.plot([lx, 5.05], [2.85, 2.85], color=C_LOOP, lw=1.6, zorder=2)          # horiz bottom
ax.plot([lx, lx],   [2.85, 6.9],  color=C_LOOP, lw=1.6, zorder=2)          # vert up
arrow(lx, 6.9, 5.05, 6.9, color=C_LOOP, lw=1.6, style="-|>")               # → ReActAgent
alabel(3.9, 4.9, "reject +\nfeedback", fs=7.5)

# ── Legend ─────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=C_ENTRY, label="API entry points"),
    mpatches.Patch(color=C_STAGE, label="Pipeline modules"),
    mpatches.Patch(color=C_TOOL,  label="ReActAgent sub-tools"),
    mpatches.Patch(color=C_EXT,   label="External services"),
    mpatches.Patch(color=C_OUT,   label="Final response"),
    mpatches.Patch(color=C_LOOP,  label="Reject / feedback loop"),
]
ax.legend(handles=handles, loc="lower left", fontsize=8,
          framealpha=0.85, edgecolor="#ccc")

plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {os.path.abspath(OUT)}")