"""
agents/showcase.py
──────────────────
Showcase Agent — Output & Human-in-the-Loop.
Owned by Devang.

Responsibilities:
  - Dispatch to the correct output format based on context.output_format
  - Wrap output generation with error handling and logging
  - Write context.final_output
  - Set context.pipeline_status → 'done'

Output formats:
  "written_review"  → output/written_review.py  (Format 1)
  "knowledge_map"   → output/knowledge_map.py   (Format 2)
  "timeline"        → output/timeline.py         (Format 3)
  "briefing"        → output/briefing.py         (Format 4, default)

The main.py orchestrator calls showcase.run(context) after analyst.run().
The Streamlit UI (ui/app.py) calls it in a background thread and polls
context.pipeline_status to update the progress display.
"""

import datetime
import json
import os
from typing import Any

from core.context import SharedContext

# Lazy imports — only load what we need
def _import_outputs():
    from output import briefing, written_review, timeline, knowledge_map
    return briefing, written_review, timeline, knowledge_map


VALID_FORMATS = {"written_review", "knowledge_map", "timeline", "briefing"}
DEFAULT_FORMAT = "briefing"


# ── Logging helper ────────────────────────────────────────────────────────────

def _log_error(
    context: SharedContext,
    stage: str,
    message: str,
    recoverable: bool = True,
) -> None:
    entry = {
        "stage": stage,
        "message": message,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "recoverable": recoverable,
    }
    context.errors.append(entry)
    level = "WARN" if recoverable else "ERROR"
    print(f"[Showcase][{stage}] {level}: {message}")


# ── Format dispatchers ────────────────────────────────────────────────────────

def _generate_briefing(context: SharedContext) -> Any:
    from output.briefing import generate
    print("[Showcase] Generating Format 4: Structured Briefing …")
    return generate(context)


def _generate_written_review(context: SharedContext) -> Any:
    from output.written_review import generate
    print("[Showcase] Generating Format 1: Written Synthesis Review …")
    return generate(context)


def _generate_timeline(context: SharedContext) -> Any:
    from output.timeline import generate
    print("[Showcase] Generating Format 3: Paradigm Timeline …")
    return generate(context)


def _generate_knowledge_map(context: SharedContext) -> Any:
    from output.knowledge_map import generate
    print("[Showcase] Generating Format 2: Knowledge Network …")
    return generate(context)


_FORMAT_DISPATCH = {
    "briefing":       _generate_briefing,
    "written_review": _generate_written_review,
    "timeline":       _generate_timeline,
    "knowledge_map":  _generate_knowledge_map,
}


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_context(context: SharedContext) -> list[str]:
    """
    Check that the context has the minimum data needed for showcase.
    Returns a list of warning strings (empty = all good).
    """
    warnings = []

    if not context.user_confirmed_papers:
        warnings.append("user_confirmed_papers is empty — no papers to showcase.")
    if not context.extracted_knowledge:
        warnings.append("extracted_knowledge is empty — Analyst may not have run.")
    if not context.output_format or context.output_format not in VALID_FORMATS:
        warnings.append(
            f"output_format '{context.output_format}' is invalid. "
            f"Defaulting to '{DEFAULT_FORMAT}'."
        )

    return warnings


# ── Main entry point ──────────────────────────────────────────────────────────

def run(context: SharedContext) -> SharedContext:
    """
    Execute the Showcase Agent and populate context.final_output.

    Reads:
        context.output_format         — which format to generate
        context.user_confirmed_papers — from HITL
        context.extracted_knowledge   — from Analyst Pass 1
        context.contradiction_report  — from Analyst Pass 3
        context.paradigm_timeline     — from Analyst Pass 4
        context.topic                 — original user topic

    Writes:
        context.final_output          — the generated output (type varies by format)
        context.pipeline_status       → 'done' | 'error'

    Args:
        context: SharedContext with analyst outputs populated.

    Returns:
        Updated SharedContext.
    """
    context.pipeline_status = "showcasing"

    print(f"\n{'='*60}")
    print(f"[Showcase] Starting output generation")
    print(f"[Showcase] Format: '{context.output_format}'")
    print(f"[Showcase] Papers: {len(context.user_confirmed_papers)}")
    print(f"[Showcase] Extractions: {len(context.extracted_knowledge)}")
    print(f"{'='*60}")

    # ── Validate ──────────────────────────────────────────────────────────────
    warnings = _validate_context(context)
    for w in warnings:
        _log_error(context, "validation", w, recoverable=True)

    # Normalize output format
    if context.output_format not in VALID_FORMATS:
        context.output_format = DEFAULT_FORMAT

    # ── Dispatch ──────────────────────────────────────────────────────────────
    generator = _FORMAT_DISPATCH[context.output_format]

    try:
        output = generator(context)
        context.final_output = output
        print(f"[Showcase] ✓ Output generated successfully ({context.output_format})")
    except Exception as exc:
        _log_error(
            context,
            f"generate_{context.output_format}",
            f"Output generation failed: {exc}",
            recoverable=False,
        )
        # Graceful degradation: fall back to briefing if primary format failed
        if context.output_format != "briefing":
            print("[Showcase] Falling back to briefing format …")
            try:
                from output.briefing import generate as gen_briefing
                context.final_output = gen_briefing(context)
                context.output_format = "briefing"
                print("[Showcase] ✓ Fallback briefing generated.")
            except Exception as fallback_exc:
                _log_error(context, "fallback_briefing", str(fallback_exc), recoverable=False)
                context.final_output = (
                    f"# Output Generation Failed\n\n"
                    f"Topic: {context.topic}\n\n"
                    f"Error: {exc}\n\nFallback error: {fallback_exc}\n\n"
                    f"Please try again or select a different output format."
                )
                context.pipeline_status = "error"
                return context
        else:
            context.final_output = (
                f"# Output Generation Failed\n\n"
                f"Topic: {context.topic}\n\nError: {exc}"
            )
            context.pipeline_status = "error"
            return context

    context.pipeline_status = "done"
    print(f"[Showcase] Pipeline complete. Status: done")
    return context


# ── Export helpers (used by UI for download buttons) ─────────────────────────

def export_as_markdown(context: SharedContext) -> str:
    """
    Return the final output as a markdown string.
    Works for written_review and briefing formats.
    For timeline/knowledge_map, returns a JSON-embedded markdown stub.
    """
    if isinstance(context.final_output, str):
        return context.final_output

    # For Plotly figures and dicts, embed metadata
    return (
        f"# {context.topic}\n\n"
        f"*Output format: {context.output_format} — see the Streamlit app for the interactive version.*\n\n"
        f"Papers analyzed: {len(context.user_confirmed_papers)}\n"
        f"Contradictions: {len(context.contradiction_report)}\n"
        f"Paradigm shifts: {len(context.paradigm_timeline)}\n"
    )


def export_as_json(context: SharedContext) -> str:
    """
    Export all analyst outputs as a structured JSON string.
    Useful for programmatic downstream consumption.
    """
    export_data = {
        "topic": context.topic,
        "output_format": context.output_format,
        "papers": [
            {
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "year": p.get("year"),
                "authors": p.get("authors", []),
                "citationCount": p.get("citationCount", 0),
                "relevance_score": p.get("relevance_score", 0.0),
                "pdf_url": p.get("pdf_url"),
            }
            for p in context.user_confirmed_papers
        ],
        "extracted_knowledge": context.extracted_knowledge,
        "contradiction_report": context.contradiction_report,
        "paradigm_timeline": context.paradigm_timeline,
        "errors": context.errors,
    }
    return json.dumps(export_data, indent=2, default=str)
