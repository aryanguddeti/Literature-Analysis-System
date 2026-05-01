"""
eval/metrics.py
───────────────
Evaluation metrics for the Literature Analysis System.
Owned by Anmol.

Four measurement areas (matching the slides):

  1. retrieval_precision_recall()
       Measures how many of the Retriever's top-N papers are genuinely relevant
       (precision) and how many of the canonical must-find papers were found
       (recall).  Target: >80% precision, >70% recall.

  2. extraction_accuracy()
       Spot-checks extracted fields (methodology, key_finding, year) against
       a manually-verified ground-truth dict for a sample of papers.
       Returns per-field accuracy and an overall score.

  3. confidence_calibration()
       Checks whether low confidence_score flags actually correspond to poor
       extractions — i.e. the model knows when it doesn't know.
       Returns a Pearson correlation between (1 - confidence) and extraction
       error rate.

  4. full_benchmark_report()
       Runs all three metrics across all three benchmark topics and prints
       a formatted summary table.  This is the main entry point for grading.

Usage
-----
    # Run the complete benchmark (all 3 topics):
    python -m eval.metrics

    # Import individual metrics in a notebook:
    from eval.metrics import retrieval_precision_recall, extraction_accuracy

All functions accept plain Python dicts/lists — no SharedContext required —
so they can be called with mock data during unit testing.
"""

import json
import math
from typing import Optional
from eval.benchmark_topics import BENCHMARK_TOPICS, TOPIC_KEYS


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(title: str) -> str:
    """Lowercase and strip punctuation for fuzzy title matching."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _title_matches(retrieved_title: str, canonical_title_lower: str) -> bool:
    """
    Returns True if retrieved_title is a close enough match to canonical_title_lower.

    Strategy: check if at least 70% of the canonical title's words appear in
    the retrieved title.  This handles slight wording differences (e.g. a
    colon vs dash in the title) without requiring exact string equality.
    """
    retrieved_norm = _normalise(retrieved_title)
    canonical_words = canonical_title_lower.split()
    if not canonical_words:
        return False
    matches = sum(1 for w in canonical_words if w in retrieved_norm)
    return (matches / len(canonical_words)) >= 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Retrieval Precision & Recall
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_precision_recall(
    retrieved_papers: list[dict],
    topic_key: str,
    top_n: int = 20,
) -> dict:
    """
    Measure retrieval quality against the benchmark canonical paper list.

    Args:
        retrieved_papers: List of paper dicts from context.deduplicated_papers
                          or context.user_confirmed_papers.  Must have a 'title'
                          key.  Pass the full list before HITL if you want to
                          evaluate the Retriever in isolation.
        topic_key:        One of TOPIC_KEYS — "attention_mechanisms", "rlhf",
                          or "graph_neural_networks".
        top_n:            How many of the retrieved papers to consider.
                          Default 20 (matches HARD_CAP).

    Returns:
        {
            "topic": str,
            "retrieved_count": int,
            "top_n": int,
            "precision": float,          # fraction of top-N that are relevant
            "recall_overall": float,     # fraction of ALL canonical papers found
            "recall_core": float,        # fraction of CORE papers found
            "found_papers": [str],       # canonical titles that were found
            "missing_papers": [str],     # canonical titles that were NOT found
            "pass_precision": bool,      # True if precision >= 0.80
            "pass_recall_core": bool,    # True if recall_core >= 0.70
        }
    """
    topic = BENCHMARK_TOPICS[topic_key]
    canonical = topic["canonical_papers"]
    core_papers = [p for p in canonical if p["importance"] == "core"]

    # Work with top-N retrieved papers only
    top_papers = retrieved_papers[:top_n]

    # ── Precision: how many retrieved papers are genuinely relevant ───────────
    # "Relevant" = appears in the canonical list (core OR supporting)
    relevant_retrieved = [
        p for p in top_papers
        if any(_title_matches(p.get("title", ""), c["title_lower"]) for c in canonical)
    ]
    precision = len(relevant_retrieved) / len(top_papers) if top_papers else 0.0

    # ── Recall: how many canonical papers were found in the retrieved set ─────
    all_retrieved_titles = [p.get("title", "") for p in retrieved_papers]

    found_papers, missing_papers = [], []
    for c in canonical:
        matched = any(_title_matches(t, c["title_lower"]) for t in all_retrieved_titles)
        if matched:
            found_papers.append(c["title_lower"])
        else:
            missing_papers.append(c["title_lower"])

    found_core = [
        c for c in core_papers
        if any(_title_matches(t, c["title_lower"]) for t in all_retrieved_titles)
    ]

    recall_overall = len(found_papers) / len(canonical) if canonical else 0.0
    recall_core = len(found_core) / len(core_papers) if core_papers else 0.0

    return {
        "topic": topic["display_name"],
        "retrieved_count": len(retrieved_papers),
        "top_n": top_n,
        "precision": round(precision, 4),
        "recall_overall": round(recall_overall, 4),
        "recall_core": round(recall_core, 4),
        "found_papers": found_papers,
        "missing_papers": missing_papers,
        "pass_precision": precision >= 0.80,
        "pass_recall_core": recall_core >= 0.70,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Extraction Accuracy
# ─────────────────────────────────────────────────────────────────────────────

# Ground-truth spot-check values for a sample of well-known papers.
# These were manually verified against the actual papers.
# Add more entries here as you manually check additional papers.
EXTRACTION_GROUND_TRUTH: dict[str, dict] = {
    # Key: lowercase title (must match _normalise output)
    "attention is all you need": {
        "year": 2017,
        "methodology_cluster": "transformer-based",
        "methodology_keywords": ["self-attention", "transformer", "multi-head"],
        "key_finding_keywords": ["translation", "state-of-the-art", "recurrence"],
        "dataset_used_keywords": ["wmt", "newstest"],
    },
    "neural machine translation by jointly learning to align and translate": {
        "year": 2015,
        "methodology_cluster": "RNN-based",
        "methodology_keywords": ["attention", "alignment", "encoder", "decoder"],
        "key_finding_keywords": ["machine translation", "alignment", "longer"],
        "dataset_used_keywords": ["wmt", "english-french", "english-german"],
    },
    "bert pre-training of deep bidirectional transformers for language understanding": {
        "year": 2019,
        "methodology_cluster": "transformer-based",
        "methodology_keywords": ["bidirectional", "pre-training", "masked", "language model"],
        "key_finding_keywords": ["fine-tuning", "eleven", "nlp", "benchmark"],
        "dataset_used_keywords": ["glue", "squad", "mnli"],
    },
    "training language models to follow instructions with human feedback": {
        "year": 2022,
        "methodology_cluster": "reinforcement-learning",
        "methodology_keywords": ["rlhf", "reward model", "ppo", "human feedback"],
        "key_finding_keywords": ["helpful", "harmless", "alignment", "prefer"],
        "dataset_used_keywords": ["labeler", "preference", "human"],
    },
    "direct preference optimization your language model is secretly a reward model": {
        "year": 2023,
        "methodology_cluster": "reinforcement-learning",
        "methodology_keywords": ["preference", "optimization", "reward", "policy"],
        "key_finding_keywords": ["rlhf", "simpler", "reward model", "equivalent"],
        "dataset_used_keywords": ["tldr", "anthropic", "hh"],
    },
    "semi-supervised classification with graph convolutional networks": {
        "year": 2017,
        "methodology_cluster": "GNN-based",
        "methodology_keywords": ["graph convolutional", "spectral", "laplacian", "node"],
        "key_finding_keywords": ["node classification", "semi-supervised", "citation"],
        "dataset_used_keywords": ["cora", "citeseer", "pubmed"],
    },
    "how powerful are graph neural networks": {
        "year": 2019,
        "methodology_cluster": "GNN-based",
        "methodology_keywords": ["weisfeiler", "expressiveness", "isomorphism", "aggregation"],
        "key_finding_keywords": ["1-wl", "gin", "expressive", "discriminative"],
        "dataset_used_keywords": ["bioinformatics", "social", "reddit"],
    },
}


def _keyword_match_score(text: str, keywords: list[str]) -> float:
    """
    Returns fraction of keywords found in text (case-insensitive).
    Score of 1.0 means all keywords present; 0.0 means none.
    """
    if not keywords or not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def extraction_accuracy(
    extracted_knowledge: list[dict],
    min_keyword_score: float = 0.5,
) -> dict:
    """
    Spot-check extracted fields against manually verified ground truth.

    Only papers whose titles appear in EXTRACTION_GROUND_TRUTH are evaluated.
    Papers not in the ground truth are skipped (not penalised).

    Args:
        extracted_knowledge: context.extracted_knowledge output from analyst.run().
        min_keyword_score:   Fraction of expected keywords that must appear for
                             a field to count as "correct". Default 0.5.

    Returns:
        {
            "papers_checked": int,
            "year_accuracy": float,
            "cluster_accuracy": float,
            "methodology_accuracy": float,
            "finding_accuracy": float,
            "overall_accuracy": float,
            "low_confidence_error_rate": float,  # errors among confidence < 0.5
            "high_confidence_error_rate": float, # errors among confidence >= 0.5
            "per_paper": [dict],                 # detail per evaluated paper
        }
    """
    per_paper_results = []

    for ext in extracted_knowledge:
        title_norm = _normalise(ext.get("title", ""))
        # Find the closest ground-truth entry
        gt = None
        for gt_title, gt_data in EXTRACTION_GROUND_TRUTH.items():
            if _title_matches(ext.get("title", ""), gt_title):
                gt = gt_data
                break
        if gt is None:
            continue   # Not in ground truth — skip

        year_correct = ext.get("year") == gt.get("year")

        extracted_cluster = (ext.get("methodology_cluster") or "").lower().strip()
        expected_cluster  = (gt.get("methodology_cluster") or "").lower().strip()
        cluster_correct   = extracted_cluster == expected_cluster

        methodology_score = _keyword_match_score(
            ext.get("methodology", ""), gt.get("methodology_keywords", [])
        )
        finding_score = _keyword_match_score(
            ext.get("key_finding", ""), gt.get("key_finding_keywords", [])
        )

        methodology_correct = methodology_score >= min_keyword_score
        finding_correct     = finding_score >= min_keyword_score

        field_scores = [year_correct, cluster_correct, methodology_correct, finding_correct]
        overall_correct = sum(field_scores) / len(field_scores)

        per_paper_results.append({
            "title": ext.get("title", ""),
            "year_correct": year_correct,
            "cluster_correct": cluster_correct,
            "methodology_correct": methodology_correct,
            "finding_correct": finding_correct,
            "overall_score": round(overall_correct, 4),
            "confidence_score": ext.get("confidence_score", 1.0),
            "had_error": overall_correct < 1.0,
        })

    if not per_paper_results:
        return {
            "papers_checked": 0,
            "year_accuracy": 0.0,
            "cluster_accuracy": 0.0,
            "methodology_accuracy": 0.0,
            "finding_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "low_confidence_error_rate": 0.0,
            "high_confidence_error_rate": 0.0,
            "per_paper": [],
        }

    n = len(per_paper_results)
    year_acc        = sum(r["year_correct"] for r in per_paper_results) / n
    cluster_acc     = sum(r["cluster_correct"] for r in per_paper_results) / n
    methodology_acc = sum(r["methodology_correct"] for r in per_paper_results) / n
    finding_acc     = sum(r["finding_correct"] for r in per_paper_results) / n
    overall_acc     = sum(r["overall_score"] for r in per_paper_results) / n

    # ── Confidence calibration preview ────────────────────────────────────────
    low_conf  = [r for r in per_paper_results if r["confidence_score"] < 0.5]
    high_conf = [r for r in per_paper_results if r["confidence_score"] >= 0.5]

    low_conf_err_rate  = sum(r["had_error"] for r in low_conf)  / len(low_conf)  if low_conf  else 0.0
    high_conf_err_rate = sum(r["had_error"] for r in high_conf) / len(high_conf) if high_conf else 0.0

    return {
        "papers_checked": n,
        "year_accuracy": round(year_acc, 4),
        "cluster_accuracy": round(cluster_acc, 4),
        "methodology_accuracy": round(methodology_acc, 4),
        "finding_accuracy": round(finding_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "low_confidence_error_rate": round(low_conf_err_rate, 4),
        "high_confidence_error_rate": round(high_conf_err_rate, 4),
        "per_paper": per_paper_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — Confidence Score Calibration
# ─────────────────────────────────────────────────────────────────────────────

def confidence_calibration(
    extracted_knowledge: list[dict],
    ground_truth: Optional[dict] = None,
) -> dict:
    """
    Measure whether the model's confidence_score is well-calibrated.

    A well-calibrated model assigns LOW confidence to papers it extracted
    poorly, and HIGH confidence to papers it extracted accurately.

    We measure this with:
      - Pearson correlation between (1 - confidence) and extraction_error
      - ECE (Expected Calibration Error) across 5 confidence bins

    Args:
        extracted_knowledge: context.extracted_knowledge from analyst.run().
        ground_truth:        Optional override for EXTRACTION_GROUND_TRUTH.

    Returns:
        {
            "papers_evaluated": int,
            "pearson_correlation": float,   # target: > 0.70 (positive = well calibrated)
            "ece": float,                   # Expected Calibration Error (lower is better)
            "calibration_bins": [dict],     # per-bin breakdown for plotting
            "is_calibrated": bool,          # True if pearson_correlation >= 0.5
        }
    """
    gt = ground_truth or EXTRACTION_GROUND_TRUTH

    # Get per-paper accuracy from extraction_accuracy()
    acc_results = extraction_accuracy(extracted_knowledge)
    evaluated = acc_results["per_paper"]

    if len(evaluated) < 3:
        return {
            "papers_evaluated": len(evaluated),
            "pearson_correlation": 0.0,
            "ece": 0.0,
            "calibration_bins": [],
            "is_calibrated": False,
            "note": "Not enough evaluated papers for calibration (need >= 3 in ground truth).",
        }

    confidences  = [r["confidence_score"] for r in evaluated]
    error_rates  = [1.0 - r["overall_score"] for r in evaluated]   # 1 = total error

    # ── Pearson correlation between (1 - confidence) and error ────────────────
    uncertainty = [1.0 - c for c in confidences]

    n = len(uncertainty)
    mean_u = sum(uncertainty) / n
    mean_e = sum(error_rates) / n

    cov = sum((u - mean_u) * (e - mean_e) for u, e in zip(uncertainty, error_rates)) / n
    std_u = math.sqrt(sum((u - mean_u) ** 2 for u in uncertainty) / n)
    std_e = math.sqrt(sum((e - mean_e) ** 2 for e in error_rates) / n)

    pearson = (cov / (std_u * std_e)) if (std_u > 0 and std_e > 0) else 0.0

    # ── Expected Calibration Error (ECE) across 5 bins ────────────────────────
    # ECE measures how closely average confidence matches average accuracy
    # across confidence buckets.
    num_bins = 5
    bins = [[] for _ in range(num_bins)]
    for conf, err in zip(confidences, error_rates):
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bins[bin_idx].append((conf, 1.0 - err))   # store (confidence, accuracy)

    ece = 0.0
    calibration_bins = []
    for i, b in enumerate(bins):
        if not b:
            calibration_bins.append({
                "bin": f"{i/num_bins:.1f}–{(i+1)/num_bins:.1f}",
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
                "gap": None,
            })
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        avg_acc  = sum(x[1] for x in b) / len(b)
        gap = abs(avg_conf - avg_acc)
        ece += (len(b) / n) * gap
        calibration_bins.append({
            "bin": f"{i/num_bins:.1f}–{(i+1)/num_bins:.1f}",
            "count": len(b),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "gap": round(gap, 4),
        })

    return {
        "papers_evaluated": n,
        "pearson_correlation": round(pearson, 4),
        "ece": round(ece, 4),
        "calibration_bins": calibration_bins,
        "is_calibrated": pearson >= 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark report
# ─────────────────────────────────────────────────────────────────────────────

def full_benchmark_report(
    topic_results: dict[str, dict],
) -> None:
    """
    Print a formatted summary table of all benchmark results.

    Args:
        topic_results: dict mapping topic_key → result dict.
                       Each result dict must have the structure returned by
                       retrieval_precision_recall().

    Example call:
        results = {}
        for key in TOPIC_KEYS:
            results[key] = retrieval_precision_recall(
                retrieved_papers=...,
                topic_key=key,
            )
        full_benchmark_report(results)
    """
    header = f"{'Topic':<35} {'Precision':>10} {'Recall(core)':>13} {'Pass P':>7} {'Pass R':>7}"
    print("\n" + "="*75)
    print("  LITERATURE ANALYSIS SYSTEM — RETRIEVAL BENCHMARK REPORT")
    print("="*75)
    print(header)
    print("─"*75)

    all_pass = True
    for key, result in topic_results.items():
        topic_name = result.get("topic", key)[:34]
        precision  = result.get("precision", 0.0)
        recall     = result.get("recall_core", 0.0)
        pass_p     = "✓" if result.get("pass_precision") else "✗"
        pass_r     = "✓" if result.get("pass_recall_core") else "✗"
        if not result.get("pass_precision") or not result.get("pass_recall_core"):
            all_pass = False
        print(f"  {topic_name:<33} {precision:>9.1%} {recall:>12.1%} {pass_p:>7} {pass_r:>7}")

    print("─"*75)
    print(f"  Targets: Precision >= 80%  |  Recall (core papers) >= 70%")
    overall = "ALL TARGETS MET ✓" if all_pass else "SOME TARGETS MISSED ✗"
    print(f"  Overall: {overall}")
    print("="*75 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point  —  python -m eval.metrics
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone runner for the benchmark.

    Because metrics.py does not call the Retriever itself, it loads paper
    lists from a saved JSON file (eval/saved_results/<topic_key>_papers.json).

    Workflow:
      1. Run the full pipeline for each benchmark topic and save results:
             from core.context import SharedContext
             import json, agents.retriever as R
             ctx = SharedContext(); ctx.topic = "attention mechanisms in NLP"
             ctx = R.run(ctx)
             with open("eval/saved_results/attention_mechanisms_papers.json","w") as f:
                 json.dump(ctx.deduplicated_papers, f)

      2. Then run:  python -m eval.metrics
    """
    import os
    import sys

    results_dir = os.path.join(os.path.dirname(__file__), "saved_results")
    os.makedirs(results_dir, exist_ok=True)

    topic_results = {}
    missing_files = []

    for key in TOPIC_KEYS:
        path = os.path.join(results_dir, f"{key}_papers.json")
        if not os.path.exists(path):
            missing_files.append(path)
            continue

        with open(path) as f:
            papers = json.load(f)

        result = retrieval_precision_recall(papers, key)
        topic_results[key] = result

        # Extraction accuracy (requires saved extracted_knowledge too)
        ext_path = os.path.join(results_dir, f"{key}_extracted.json")
        if os.path.exists(ext_path):
            with open(ext_path) as f:
                extracted = json.load(f)
            acc = extraction_accuracy(extracted)
            cal = confidence_calibration(extracted)
            print(f"\n[{key}] Extraction accuracy  : {acc['overall_accuracy']:.1%}")
            print(f"[{key}] Confidence calibration: pearson={cal['pearson_correlation']:.2f}  "
                  f"ECE={cal['ece']:.3f}  calibrated={cal['is_calibrated']}")

    if topic_results:
        full_benchmark_report(topic_results)

    if missing_files:
        print("The following result files were not found — run the pipeline first:")
        for p in missing_files:
            print(f"  {p}")
        if not topic_results:
            print("\nNo results to show. See the docstring in this file for instructions.")
            sys.exit(1)
