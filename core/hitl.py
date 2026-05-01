"""
core/hitl.py
────────────
Human-in-the-Loop checkpoint logic.
Owned by Devang (Showcase/UI).

Manages the pause between Retriever and Analyst:
  - Presents deduplicated_papers to the user for confirmation
  - Writes user_confirmed_papers to SharedContext
  - Updates pipeline_status accordingly

Two modes:
  1. Streamlit mode  — called by ui/app.py; uses session_state flags
  2. Headless mode   — called by main.py CLI; auto-confirms all papers
     (useful for testing without a browser)
"""

from __future__ import annotations
import datetime
from core.context import SharedContext


# ── Status helpers ────────────────────────────────────────────────────────────

def is_awaiting_hitl(context: SharedContext) -> bool:
    """Return True if the pipeline is paused at the HITL checkpoint."""
    return context.pipeline_status == "awaiting_hitl"


def confirm_papers(
    context: SharedContext,
    selected_paper_ids: list[str],
) -> SharedContext:
    """
    Record the user's paper selection and advance the pipeline.

    Args:
        context:            SharedContext with deduplicated_papers populated.
        selected_paper_ids: List of paperIds the user confirmed at the HITL screen.

    Returns:
        Updated context with user_confirmed_papers set and
        pipeline_status → 'analysing'.
    """
    id_set = set(selected_paper_ids)
    confirmed = [
        p for p in context.deduplicated_papers
        if (p.get("paperId") or p.get("arxiv_id", "")) in id_set
    ]

    if not confirmed:
        context.errors.append({
            "stage": "hitl",
            "message": "User confirmed zero papers — cannot proceed with analysis.",
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "recoverable": False,
        })
        context.pipeline_status = "error"
        return context

    context.user_confirmed_papers = confirmed
    context.pipeline_status = "analysing"
    print(
        f"[HITL] User confirmed {len(confirmed)} / "
        f"{len(context.deduplicated_papers)} papers."
    )
    return context


def auto_confirm_all(context: SharedContext) -> SharedContext:
    """
    Headless mode: confirm all deduplicated papers without user interaction.
    Used by main.py for CLI / testing runs.
    """
    all_ids = [
        p.get("paperId") or p.get("arxiv_id", "")
        for p in context.deduplicated_papers
    ]
    return confirm_papers(context, all_ids)


# ── Paper display helpers (used by Streamlit UI) ──────────────────────────────

def format_paper_for_display(paper: dict) -> dict:
    """
    Return a display-friendly version of a paper dict for the HITL screen.
    Truncates abstract for readability.
    """
    abstract = paper.get("abstract", "") or ""
    authors = paper.get("authors", []) or []

    return {
        "paperId":        paper.get("paperId") or paper.get("arxiv_id", ""),
        "title":          paper.get("title", "Untitled"),
        "year":           paper.get("year") or "N/A",
        "authors":        ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
        "abstract_short": abstract[:300] + "…" if len(abstract) > 300 else abstract,
        "relevance_score": round(paper.get("relevance_score", 0.0), 3),
        "citationCount":  paper.get("citationCount", 0),
        "source":         paper.get("source", "unknown"),
        "pdf_url":        paper.get("pdf_url"),
    }


def get_display_papers(context: SharedContext) -> list[dict]:
    """
    Return all deduplicated papers formatted for the HITL checklist UI.
    Sorted by relevance_score descending (Retriever already does this,
    but we re-sort defensively).
    """
    papers = sorted(
        context.deduplicated_papers,
        key=lambda p: p.get("relevance_score", 0.0),
        reverse=True,
    )
    return [format_paper_for_display(p) for p in papers]
