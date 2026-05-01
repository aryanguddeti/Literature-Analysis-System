"""
main.py
───────
CLI orchestrator for the Literature Analysis System.
Owned by Devang (wires all 3 agents together).

Usage:
    python main.py --topic "attention mechanisms in NLP"
    python main.py --topic "RLHF" --format briefing
    python main.py --topic "graph neural networks" --format written_review --no-hitl

Flags:
    --topic    Research topic (required)
    --format   Output format: briefing | written_review | timeline | knowledge_map
               Default: briefing
    --no-hitl  Skip HITL confirmation (auto-confirm all papers). Useful for testing.
    --out      Output file path (default: output_<topic>.md or .json)
"""

import argparse
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from core.context import SharedContext
from core.hitl import auto_confirm_all
import agents.retriever as retriever_agent
import agents.analyst as analyst_agent
import agents.showcase as showcase_agent

VALID_FORMATS = {"briefing", "written_review", "timeline", "knowledge_map"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Literature Analysis System — CLI runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--topic", required=True, help="Research topic to analyze")
    parser.add_argument(
        "--format",
        default="briefing",
        choices=sorted(VALID_FORMATS),
        help="Output format (default: briefing)",
    )
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Auto-confirm all papers (skip HITL checkpoint)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path (auto-named if not provided)",
    )
    return parser.parse_args()


def _print_banner(topic: str, fmt: str):
    print("\n" + "=" * 60)
    print("  Literature Analysis System")
    print("=" * 60)
    print(f"  Topic:  {topic}")
    print(f"  Format: {fmt}")
    print("=" * 60 + "\n")


def _print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _hitl_cli(context: SharedContext) -> SharedContext:
    """
    CLI HITL checkpoint: display papers and ask user to confirm.
    """
    papers = context.deduplicated_papers
    _print_section(f"HITL Checkpoint — {len(papers)} papers retrieved")

    for i, p in enumerate(papers, 1):
        relevance = p.get("relevance_score", 0.0)
        citations = p.get("citationCount", 0)
        year = p.get("year", "N/A")
        print(f"  {i:>2}. [{relevance:.2f}] {p.get('title', '?')[:65]}")
        print(f"       {year} · {citations:,} citations · {p.get('source', '?')}")

    print(f"\n  Enter paper numbers to include (e.g. 1,2,5-8,10)")
    print(f"  Or press Enter to include ALL {len(papers)} papers.")
    print(f"  Type 'q' to quit.\n")

    raw = input("  Your selection: ").strip()

    if raw.lower() == "q":
        print("Aborted.")
        sys.exit(0)

    if not raw:
        return auto_confirm_all(context)

    # Parse selection (handles: "1,3,5-8")
    selected_ids = []
    parts = raw.replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            try:
                start, end = part.split("-")
                for idx in range(int(start), int(end) + 1):
                    if 1 <= idx <= len(papers):
                        pid = papers[idx - 1].get("paperId") or papers[idx - 1].get("arxiv_id", "")
                        if pid:
                            selected_ids.append(pid)
            except ValueError:
                pass
        else:
            try:
                idx = int(part)
                if 1 <= idx <= len(papers):
                    pid = papers[idx - 1].get("paperId") or papers[idx - 1].get("arxiv_id", "")
                    if pid:
                        selected_ids.append(pid)
            except ValueError:
                pass

    if not selected_ids:
        print("  No valid selection — auto-confirming all papers.")
        return auto_confirm_all(context)

    from core.hitl import confirm_papers
    return confirm_papers(context, selected_ids)


def _save_output(context: SharedContext, output_path: str | None) -> str:
    """Save final output to file. Returns the path written."""
    topic_slug = context.topic[:30].replace(" ", "_").replace("/", "_")

    if context.output_format in ("briefing", "written_review"):
        path = output_path or f"output_{topic_slug}.md"
        content = showcase_agent.export_as_markdown(context)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        path = output_path or f"output_{topic_slug}.json"
        content = showcase_agent.export_as_json(context)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    return path


def main():
    args = parse_args()
    _print_banner(args.topic, args.format)

    context = SharedContext(
        topic=args.topic,
        output_format=args.format,
    )

    t0 = time.time()

    # ── Step 1: Retriever ─────────────────────────────────────────────────────
    _print_section("Step 1/3 — Retriever")
    context = retriever_agent.run(context)

    if context.pipeline_status == "error":
        errors = [e for e in context.errors if not e.get("recoverable")]
        msg = errors[-1]["message"] if errors else "Retrieval failed."
        print(f"\n[FATAL] Retriever error: {msg}")
        sys.exit(1)

    print(f"\n[OK] Retrieved {len(context.deduplicated_papers)} deduplicated papers.")

    # ── Step 2: HITL ──────────────────────────────────────────────────────────
    _print_section("Step 2/3 — HITL Checkpoint")

    if args.no_hitl:
        print("  --no-hitl flag set: auto-confirming all papers.")
        context = auto_confirm_all(context)
    else:
        context = _hitl_cli(context)

    if context.pipeline_status == "error":
        print("[FATAL] HITL step failed.")
        sys.exit(1)

    print(f"\n[OK] {len(context.user_confirmed_papers)} papers confirmed for analysis.")

    # ── Step 3: Analyst ───────────────────────────────────────────────────────
    _print_section("Step 3/3 — Analyst")
    context = analyst_agent.run(context)

    if context.pipeline_status == "error":
        errors = [e for e in context.errors if not e.get("recoverable")]
        msg = errors[-1]["message"] if errors else "Analysis failed."
        print(f"\n[FATAL] Analyst error: {msg}")
        sys.exit(1)

    print(f"\n[OK] Analysis complete.")
    print(f"     Extractions: {len(context.extracted_knowledge)}")
    print(f"     Contradictions: {len(context.contradiction_report)}")
    print(f"     Paradigm shifts: {len(context.paradigm_timeline)}")

    # ── Step 4: Showcase ──────────────────────────────────────────────────────
    _print_section("Step 4/3 — Showcase")
    context = showcase_agent.run(context)

    if context.pipeline_status == "error":
        print("[WARN] Showcase failed — saving JSON fallback.")
        context.output_format = "json"

    # ── Save output ───────────────────────────────────────────────────────────
    path = _save_output(context, args.out)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  ✓ Done in {elapsed:.1f}s")
    print(f"  Output saved to: {path}")
    print(f"  Format: {context.output_format}")

    if context.errors:
        print(f"\n  Warnings ({len(context.errors)}):")
        for e in context.errors:
            icon = "🔴" if not e.get("recoverable") else "🟡"
            print(f"    {icon} [{e['stage']}] {e['message'][:80]}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
