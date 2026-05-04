"""
eval/benchmark_runner.py
────────────────────────
Evaluation benchmark runner for the Literature Analysis System.
Owned by Devang.

Runs the full pipeline on all 3 benchmark topics and measures:
  - Synthesis quality rubric score (Devang's metric)
  - End-to-end runtime

Usage:
    python eval/benchmark_runner.py
    python eval/benchmark_runner.py --topic attention_mechanisms
    python eval/benchmark_runner.py --all --save-results
"""

import argparse
import json
import time
import datetime
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.context import SharedContext
from core.hitl import auto_confirm_all
from eval.benchmark_topics import BENCHMARK_TOPICS, TOPIC_KEYS
import agents.retriever as retriever_agent
import agents.analyst as analyst_agent
import agents.showcase as showcase_agent


# ── Synthesis Quality Rubric ──────────────────────────────────────────────────
# Devang's metric: score the quality of the generated output on 5 dimensions.
# Each dimension scored 0-5. Total max = 25. Normalized to 0-5 average.

RUBRIC_DIMENSIONS = [
    "thematic_organization",   # Is the output organized by theme/cluster?
    "citation_accuracy",       # Are papers cited correctly with year/author?
    "contradiction_coverage",  # Are contradictions mentioned and explained?
    "clarity",                 # Is the writing clear and readable?
    "completeness",            # Does it cover all major papers in the set?
]


def _score_output_automatically(context: SharedContext, topic_key: str) -> dict:
    """
    Automatically score the output quality against the benchmark topic.
    Returns a dict of dimension scores and overall average.

    This is a heuristic scorer — for human eval, use the rubric manually.
    """
    benchmark = BENCHMARK_TOPICS[topic_key]
    output = context.final_output or ""
    output_lower = output.lower() if isinstance(output, str) else ""

    scores = {}

    # 1. Thematic organization — check if cluster names appear in output
    expected_clusters = benchmark.get("expected_clusters", [])
    clusters_found = sum(1 for c in expected_clusters if c.lower() in output_lower)
    scores["thematic_organization"] = round((clusters_found / max(len(expected_clusters), 1)) * 5, 1)

    # 2. Citation accuracy — check if canonical paper titles appear
    canonical = benchmark.get("canonical_papers", [])
    core_papers = [p for p in canonical if p["importance"] == "core"]
    confirmed_titles = [p.get("title", "").lower() for p in context.user_confirmed_papers]
    papers_cited = sum(
        1 for cp in core_papers
        if any(cp["title_lower"][:20] in t for t in confirmed_titles)
    )
    scores["citation_accuracy"] = round((papers_cited / max(len(core_papers), 1)) * 5, 1)

    # 3. Contradiction coverage — check if contradictions are in output
    if context.contradiction_report:
        major = [c for c in context.contradiction_report if c.get("severity") == "major"]
        scores["contradiction_coverage"] = 5.0 if major else 3.0
    else:
        scores["contradiction_coverage"] = 1.0 if "no contradiction" in output_lower else 0.0

    # 4. Clarity — proxy: output length and structure
    if isinstance(output, str):
        word_count = len(output.split())
        has_sections = output.count("##") >= 3
        scores["clarity"] = min(5.0, round((word_count / 200) + (2 if has_sections else 0), 1))
    else:
        scores["clarity"] = 3.0  # Default for non-text outputs

    # 5. Completeness — how many confirmed papers vs expected
    confirmed = len(context.user_confirmed_papers)
    scores["completeness"] = min(5.0, round((confirmed / 10) * 5, 1))

    average = round(sum(scores.values()) / len(scores), 2)
    return {"dimensions": scores, "average": average, "max": 5.0}


def run_single_topic(
    topic_key: str,
    output_format: str = "briefing",
    auto_hitl: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the full pipeline on a single benchmark topic and return results.

    Args:
        topic_key:     Key from BENCHMARK_TOPICS (e.g. 'attention_mechanisms')
        output_format: Output format to generate
        auto_hitl:     Auto-confirm all papers (True for automated runs)
        verbose:       Print progress to console

    Returns:
        Dict with timing, scores, errors, and context summary.
    """
    benchmark = BENCHMARK_TOPICS[topic_key]
    topic_query = benchmark["query"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running benchmark: {benchmark['display_name']}")
        print(f"Query: {topic_query}")
        print(f"{'='*60}")

    result = {
        "topic_key": topic_key,
        "display_name": benchmark["display_name"],
        "output_format": output_format,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "timing": {},
        "pipeline_status": None,
        "paper_counts": {},
        "quality_scores": {},
        "errors": [],
    }

    context = SharedContext(topic=topic_query, output_format=output_format)
    t_start = time.time()

    # ── Retriever ─────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        context = retriever_agent.run(context)
    except Exception as e:
        result["errors"].append({"stage": "retriever", "error": str(e)})
    result["timing"]["retriever_s"] = round(time.time() - t0, 1)

    if context.pipeline_status == "error":
        result["pipeline_status"] = "error"
        result["errors"] += context.errors
        return result

    # ── HITL ──────────────────────────────────────────────────────────────────
    if auto_hitl:
        context = auto_confirm_all(context)

    result["paper_counts"]["retrieved"] = len(context.deduplicated_papers)
    result["paper_counts"]["confirmed"] = len(context.user_confirmed_papers)

    # ── Analyst ───────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        context = analyst_agent.run(context)
    except Exception as e:
        result["errors"].append({"stage": "analyst", "error": str(e)})
    result["timing"]["analyst_s"] = round(time.time() - t0, 1)

    result["paper_counts"]["extracted"] = len(context.extracted_knowledge)
    result["paper_counts"]["contradictions"] = len(context.contradiction_report)
    result["paper_counts"]["paradigm_shifts"] = len(context.paradigm_timeline)

    # ── Showcase ──────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        context = showcase_agent.run(context)
    except Exception as e:
        result["errors"].append({"stage": "showcase", "error": str(e)})
    result["timing"]["showcase_s"] = round(time.time() - t0, 1)

    # ── Total runtime ─────────────────────────────────────────────────────────
    result["timing"]["total_s"] = round(time.time() - t_start, 1)
    result["pipeline_status"] = context.pipeline_status
    result["errors"] += [e for e in context.errors if not e.get("recoverable")]

    # ── Quality scoring ───────────────────────────────────────────────────────
    try:
        result["quality_scores"] = _score_output_automatically(context, topic_key)
    except Exception as e:
        result["quality_scores"] = {"error": str(e)}

    if verbose:
        print(f"\n[Result] Status: {result['pipeline_status']}")
        print(f"[Result] Papers: {result['paper_counts']}")
        print(f"[Result] Timing: {result['timing']}")
        if result["quality_scores"].get("average"):
            print(f"[Result] Quality: {result['quality_scores']['average']:.2f}/5.0")

    return result


def run_all_topics(output_format: str = "briefing", save_results: bool = False) -> list[dict]:
    """
    Run the benchmark on all 3 topics and aggregate results.

    Args:
        output_format: Output format to use for all topics
        save_results:  Save results to eval/results.json

    Returns:
        List of result dicts, one per topic.
    """
    print(f"\n{'='*60}")
    print(f"  Literature Analysis System — Evaluation Benchmark")
    print(f"  Topics: {len(TOPIC_KEYS)} | Format: {output_format}")
    print(f"{'='*60}")

    all_results = []

    for topic_key in TOPIC_KEYS:
        result = run_single_topic(topic_key, output_format=output_format)
        all_results.append(result)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")

    total_time = sum(r["timing"].get("total_s", 0) for r in all_results)
    avg_quality = sum(
        r["quality_scores"].get("average", 0)
        for r in all_results
        if "average" in r.get("quality_scores", {})
    ) / len(all_results)

    print(f"  Total runtime:     {total_time:.1f}s")
    print(f"  Avg quality score: {avg_quality:.2f}/5.0")
    print(f"\n  Per-topic results:")

    for r in all_results:
        status_icon = "✓" if r["pipeline_status"] == "done" else "✗"
        quality = r["quality_scores"].get("average", 0)
        print(
            f"  {status_icon} {r['display_name'][:35]:<35} "
            f"Quality: {quality:.2f}/5.0  "
            f"Time: {r['timing'].get('total_s', 0):.0f}s"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    if save_results:
        results_path = ROOT / "eval" / "results.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {results_path}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark")
    parser.add_argument("--topic", choices=TOPIC_KEYS, help="Run single topic")
    parser.add_argument("--all", action="store_true", help="Run all topics")
    parser.add_argument("--format", default="briefing", choices=["briefing", "written_review", "timeline", "knowledge_map"])
    parser.add_argument("--save-results", action="store_true", help="Save to eval/results.json")
    args = parser.parse_args()

    if args.topic:
        run_single_topic(args.topic, output_format=args.format)
    else:
        run_all_topics(output_format=args.format, save_results=args.save_results)


if __name__ == "__main__":
    main()
