"""
agents/analyst.py
─────────────────
Analyst Agent — Knowledge Extraction & Reasoning.
Owned by Anmol.

Pipeline (run in order after HITL checkpoint):
  1. extract_paper_knowledge   — Pass 1: per-paper JSON extraction (methodology,
                                 finding, dataset, metric, confidence_score)
  2. cluster_methodologies     — Pass 2: group papers into high-level approach
                                 categories using pre-computed embeddings
  3. detect_contradictions     — Pass 3: pairwise comparison within each cluster
  4. build_paradigm_timeline   — Pass 4 (Historian): identify paradigm shifts
                                 sorted chronologically across all papers

On completion, sets:
  context.extracted_knowledge   (list[dict])
  context.contradiction_report  (list[dict])
  context.paradigm_timeline     (list[dict])
  context.pipeline_status       → 'showcasing'

On unrecoverable error, sets context.pipeline_status → 'error'.

Integration notes from Retriever review:
  - context.paper_embeddings already populated (Gemini gemini-embedding-001,
    dim=768). Do NOT recompute — read from context directly.
  - HARD_CAP is 20 papers, not 30 as originally planned. Loops sized accordingly.
  - Papers are pre-sorted by relevance_score descending.
  - Each paper dict keys: paperId, title, abstract, year, authors, citationCount,
    references, source, arxiv_id, pdf_url, relevance_score
"""

import os
import json
import time
import datetime
import numpy as np
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from core.context import SharedContext
from core.embeddings import cosine_similarity, EMBEDDING_DIM

load_dotenv()

# ── LLM configuration ─────────────────────────────────────────────────────────
# Using Gemini 2.5 Flash — fast, large context window, free-tier accessible
# paper abstracts in a single call to stay within rate limits.

_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,          # Low temp for structured extraction (deterministic)
    max_tokens=8192,
    timeout=60,
    max_retries=3,
)

# ── Tuning constants ──────────────────────────────────────────────────────────

EXTRACTION_BATCH_SIZE = 5     # Papers per Gemini call (stay within rate limits)
INTER_BATCH_DELAY     = 1.0   # Seconds between extraction batches
MIN_CLUSTER_SIZE      = 2     # Clusters smaller than this are merged into "Other"
CONTRADICTION_DELAY   = 0.5   # Seconds between contradiction detection calls
MIN_CONFIDENCE        = 0.3


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_error(
    context: SharedContext,
    stage: str,
    message: str,
    recoverable: bool = True,
) -> None:
    """Append a structured error to context.errors — mirrors Retriever pattern."""
    entry = {
        "stage": stage,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "recoverable": recoverable,
    }
    context.errors.append(entry)
    level = "WARN" if recoverable else "ERROR"
    print(f"[Analyst][{stage}] {level}: {message}")


def _call_llm(prompt: str, stage: str) -> Optional[str]:
    """
    Single LLM call with error handling.
    Returns raw text response or None on failure.
    """
    try:
        response = _LLM.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:
        print(f"[Analyst][{stage}] LLM call failed: {exc}")
        return None


def _parse_json_safe(raw: str, stage: str) -> Optional[any]:
    """
    Strip markdown fences and parse JSON safely.
    Returns parsed object or None on failure.
    """
    if not raw:
        return None
    # Strip ```json ... ``` fences that Gemini sometimes adds
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"[Analyst][{stage}] JSON parse error: {exc}")
        print(f"[Analyst][{stage}] Raw response (first 300 chars): {raw[:300]}")
        return None


# ── Pass 1: Per-paper JSON extraction ────────────────────────────────────────

_EXTRACTION_PROMPT = """You are a research analyst extracting structured metadata from academic paper abstracts.

Analyze the following {n} paper(s) and return a JSON array with exactly {n} objects.
Each object must have these exact keys:

{{
  "paperId": "<copy from input>",
  "title": "<copy from input>",
  "year": <integer or null>,
  "methodology": "<1-2 sentence description of the core technique or approach used>",
  "methodology_cluster": "<ONE of: transformer-based | reinforcement-learning | GNN-based | CNN-based | RNN-based | optimization | probabilistic | hybrid | survey | other>",
  "key_finding": "<1-2 sentence description of the main result or contribution>",
  "dataset_used": "<name of primary dataset/benchmark, or 'not specified'>",
  "reported_metric": "<main evaluation metric and value, e.g. 'BLEU 41.0', or 'not specified'>",
  "baseline_compared": ["<list of baseline methods compared against, empty list if none>"],
  "confidence_score": <float 0.0-1.0 reflecting your confidence in this extraction>,
  "low_confidence_reason": "<brief reason if confidence < 0.5, else empty string>"
}}

IMPORTANT:
- Return ONLY the JSON array. No explanation, no markdown fences.
- If the abstract lacks enough information for a field, use "not specified" (strings) or null (numbers).
- confidence_score should be LOW (<0.5) if: abstract is missing or very short, methodology is ambiguous,
  or the paper is a survey with no single methodology.

Papers to analyze:
{papers_json}
"""


def _extract_batch(papers: list[dict], stage: str = "extraction") -> list[dict]:
    """
    Run Pass 1 extraction on a batch of papers in a single LLM call.

    Args:
        papers: List of paper dicts (must have paperId, title, abstract, year).

    Returns:
        List of extracted knowledge dicts. Falls back to skeleton dicts on failure.
    """
    # Prepare a slim version of each paper for the prompt (no embeddings, no references)
    slim = [
        {
            "paperId": p.get("paperId", ""),
            "title": p.get("title", ""),
            "abstract": (p.get("abstract") or "")[:1500],  # Truncate long abstracts
            "year": p.get("year"),
        }
        for p in papers
    ]

    prompt = _EXTRACTION_PROMPT.format(
        n=len(papers),
        papers_json=json.dumps(slim, indent=2),
    )

    raw = _call_llm(prompt, stage)
    parsed = _parse_json_safe(raw, stage) if raw else None

    if parsed and isinstance(parsed, list) and len(parsed) == len(papers):
        return parsed

    # Fallback: return skeleton dicts so the pipeline can continue
    print(f"[Analyst][{stage}] Batch extraction failed — using skeleton fallback for {len(papers)} papers.")
    return [
        {
            "paperId": p.get("paperId", ""),
            "title": p.get("title", ""),
            "year": p.get("year"),
            "methodology": "extraction failed",
            "methodology_cluster": "other",
            "key_finding": "extraction failed",
            "dataset_used": "not specified",
            "reported_metric": "not specified",
            "baseline_compared": [],
            "confidence_score": 0.0,
            "low_confidence_reason": "LLM extraction failed or returned malformed JSON",
        }
        for p in papers
    ]


def extract_paper_knowledge(papers: list[dict]) -> list[dict]:
    """
    Pass 1: Extract structured knowledge from all papers in batches.

    Uses EXTRACTION_BATCH_SIZE papers per LLM call to balance cost and
    rate limits. Pauses INTER_BATCH_DELAY seconds between batches.

    Args:
        papers: List of paper dicts from context.user_confirmed_papers.

    Returns:
        List of extracted knowledge dicts, one per paper, in the same order.
    """
    print(f"\n[Analyst] Pass 1: Extracting knowledge from {len(papers)} papers …")
    all_extractions: list[dict] = []

    for i in range(0, len(papers), EXTRACTION_BATCH_SIZE):
        batch = papers[i : i + EXTRACTION_BATCH_SIZE]
        batch_num = i // EXTRACTION_BATCH_SIZE + 1
        total_batches = (len(papers) + EXTRACTION_BATCH_SIZE - 1) // EXTRACTION_BATCH_SIZE

        print(f"[Analyst]   Batch {batch_num}/{total_batches} ({len(batch)} papers)")
        extractions = _extract_batch(batch, stage=f"extraction_batch_{batch_num}")
        all_extractions.extend(extractions)

        # Flag low-confidence extractions for the UI
        for ext in extractions:
            score = ext.get("confidence_score", 0.0)
            if score < MIN_CONFIDENCE:
                print(
                    f"[Analyst]   ⚠ Low confidence ({score:.2f}): '{ext.get('title', '?')[:50]}'"
                    f" — {ext.get('low_confidence_reason', '')}"
                )

        if i + EXTRACTION_BATCH_SIZE < len(papers):
            time.sleep(INTER_BATCH_DELAY)

    print(f"[Analyst] Pass 1 complete — {len(all_extractions)} papers extracted.")
    avg_confidence = (
        sum(e.get("confidence_score", 0.0) for e in all_extractions) / len(all_extractions)
        if all_extractions else 0.0
    )
    print(f"[Analyst]   Average confidence score: {avg_confidence:.2f}")
    return all_extractions


# ── Pass 2: Methodology clustering ───────────────────────────────────────────

def cluster_methodologies(
    extractions: list[dict],
    paper_embeddings: dict[str, list[float]],
) -> dict[str, list[dict]]:
    """
    Pass 2: Group extracted papers into methodology clusters.

    Strategy:
      - Primary: use methodology_cluster label from Pass 1 (Gemini-assigned).
      - Secondary: if a cluster has fewer than MIN_CLUSTER_SIZE papers,
        reassign its members to the nearest larger cluster using cosine
        similarity on the pre-computed paper embeddings from SharedContext.
        This avoids tiny clusters that make contradiction detection meaningless.

    Args:
        extractions:      Output of extract_paper_knowledge().
        paper_embeddings: context.paper_embeddings (paperId → embedding vector).
                          Pre-computed by Retriever — do NOT recompute.

    Returns:
        Dict mapping cluster_name → list of extraction dicts in that cluster.
    """
    print(f"\n[Analyst] Pass 2: Clustering {len(extractions)} papers by methodology …")

    # ── Initial clustering by LLM-assigned label ──────────────────────────────
    raw_clusters: dict[str, list[dict]] = {}
    for ext in extractions:
        cluster = ext.get("methodology_cluster") or "other"
        raw_clusters.setdefault(cluster, []).append(ext)

    print(f"[Analyst]   Initial clusters: { {k: len(v) for k, v in raw_clusters.items()} }")

    # ── Merge small clusters into nearest large cluster ───────────────────────
    large_clusters = {k: v for k, v in raw_clusters.items() if len(v) >= MIN_CLUSTER_SIZE}
    small_clusters = {k: v for k, v in raw_clusters.items() if len(v) < MIN_CLUSTER_SIZE}

    if not small_clusters:
        print(f"[Analyst]   No small clusters to merge.")
        return raw_clusters

    if not large_clusters:
        # All clusters are small — keep as-is, merge everything into 'other'
        print(f"[Analyst]   All clusters small — merging into 'other'.")
        merged = {"other": extractions}
        return merged

    # Compute centroid embedding for each large cluster
    cluster_centroids: dict[str, list[float]] = {}
    for cluster_name, members in large_clusters.items():
        member_embeddings = []
        for m in members:
            pid = m.get("paperId", "")
            emb = paper_embeddings.get(pid)
            if emb:
                member_embeddings.append(emb)
        if member_embeddings:
            centroid = np.mean(member_embeddings, axis=0).tolist()
            cluster_centroids[cluster_name] = centroid

    # Reassign each small-cluster paper to the closest large cluster
    final_clusters = {k: list(v) for k, v in large_clusters.items()}

    for cluster_name, members in small_clusters.items():
        for paper in members:
            pid = paper.get("paperId", "")
            emb = paper_embeddings.get(pid)

            if emb and cluster_centroids:
                # Find the large cluster whose centroid is most similar
                best_cluster = max(
                    cluster_centroids,
                    key=lambda cn: cosine_similarity(emb, cluster_centroids[cn]),
                )
                final_clusters[best_cluster].append(paper)
                print(
                    f"[Analyst]   Merged '{paper.get('title', '?')[:40]}' "
                    f"({cluster_name}) → {best_cluster}"
                )
            else:
                # No embedding available — fall back to 'other'
                final_clusters.setdefault("other", []).append(paper)

    print(f"[Analyst]   Final clusters: { {k: len(v) for k, v in final_clusters.items()} }")
    return final_clusters


# ── Pass 3: Contradiction detection ──────────────────────────────────────────

_CONTRADICTION_PROMPT = """You are a critical research analyst. Compare these two research papers and determine if they make contradictory claims.

Paper A:
  Title: {title_a}
  Year: {year_a}
  Key finding: {finding_a}
  Methodology: {methodology_a}

Paper B:
  Title: {title_b}
  Year: {year_b}
  Key finding: {finding_b}
  Methodology: {methodology_b}

Return ONLY a JSON object with these exact keys:
{{
  "contradiction_found": <true or false>,
  "claim_a": "<the specific claim from Paper A that conflicts, or empty string if no contradiction>",
  "claim_b": "<the specific claim from Paper B that conflicts, or empty string if no contradiction>",
  "reasoning": "<1-2 sentence explanation of why this is/isn't a contradiction>",
  "severity": "<'major' | 'minor' | 'none'>"
}}

A contradiction is 'major' if the papers directly disagree on a factual claim, benchmark result,
or core conclusion. It is 'minor' if the disagreement is about scope, dataset, or methodology choice.
It is 'none' if the papers simply address different aspects or use different approaches without conflicting.
"""


def detect_contradictions(
    clusters: dict[str, list[dict]],
) -> list[dict]:
    """
    Pass 3: Detect contradictions between papers within the same cluster.

    Only compares pairs within the same methodology cluster — comparing papers
    from entirely different approaches (e.g. transformer vs GNN) rarely yields
    meaningful contradictions, and doing all-pairs across 20 papers = 190 calls.

    Within a cluster of N papers: N*(N-1)/2 pairs.
    For N=5 (typical): 10 pairs per cluster. Manageable.

    Args:
        clusters: Output of cluster_methodologies().

    Returns:
        List of contradiction dicts with paper references and reasoning.
        Only includes entries where contradiction_found=True.
    """
    print(f"\n[Analyst] Pass 3: Detecting contradictions across {len(clusters)} clusters …")
    contradictions: list[dict] = []
    total_pairs_checked = 0

    for cluster_name, papers in clusters.items():
        if len(papers) < 2:
            continue  # Can't have a contradiction with one paper

        print(f"[Analyst]   Checking cluster '{cluster_name}' ({len(papers)} papers) …")

        for i, pa in enumerate(papers):
            for pb in papers[i + 1 :]:
                total_pairs_checked += 1

                prompt = _CONTRADICTION_PROMPT.format(
                    title_a=pa.get("title", "?"),
                    year_a=pa.get("year", "?"),
                    finding_a=pa.get("key_finding", "not specified"),
                    methodology_a=pa.get("methodology", "not specified"),
                    title_b=pb.get("title", "?"),
                    year_b=pb.get("year", "?"),
                    finding_b=pb.get("key_finding", "not specified"),
                    methodology_b=pb.get("methodology", "not specified"),
                )

                raw = _call_llm(prompt, f"contradiction_{cluster_name}")
                parsed = _parse_json_safe(raw, f"contradiction_{cluster_name}") if raw else None

                if parsed and parsed.get("contradiction_found"):
                    entry = {
                        "paper_a_id":    pa.get("paperId", ""),
                        "paper_a_title": pa.get("title", ""),
                        "paper_a_year":  pa.get("year"),
                        "paper_b_id":    pb.get("paperId", ""),
                        "paper_b_title": pb.get("title", ""),
                        "paper_b_year":  pb.get("year"),
                        "cluster":       cluster_name,
                        "claim_a":       parsed.get("claim_a", ""),
                        "claim_b":       parsed.get("claim_b", ""),
                        "reasoning":     parsed.get("reasoning", ""),
                        "severity":      parsed.get("severity", "minor"),
                    }
                    contradictions.append(entry)
                    print(
                        f"[Analyst]   ⚡ {parsed.get('severity','?').upper()} contradiction: "
                        f"'{pa.get('title','?')[:35]}' vs '{pb.get('title','?')[:35]}'"
                    )

                time.sleep(CONTRADICTION_DELAY)

    print(
        f"[Analyst] Pass 3 complete — {len(contradictions)} contradictions found "
        f"across {total_pairs_checked} pairs checked."
    )
    return contradictions


# ── Pass 4: Paradigm shift timeline (Historian pass) ─────────────────────────

_TIMELINE_PROMPT = """You are a research historian. Analyze the following papers sorted by year and identify significant paradigm shifts and milestones in this research area.

Topic: {topic}

Papers (sorted by year):
{papers_json}

Return a JSON array of paradigm shift events. Each event must have:
{{
  "year": <integer>,
  "shift_description": "<1-2 sentence description of what changed and why it mattered>",
  "key_paper": "<title of the paper that best represents this shift>",
  "key_paper_id": "<paperId of the key paper>",
  "impact": "<'incremental' | 'significant' | 'paradigm-shift'>",
  "affected_cluster": "<the methodology cluster most affected by this shift>"
}}

Guidelines:
- A 'paradigm-shift' is a fundamental change in how the field approaches the problem
  (e.g. introduction of transformers replaced RNNs for sequence modeling).
- A 'significant' event is an important improvement that doesn't replace the prior approach.
- An 'incremental' event is a steady improvement or refinement.
- Only include events that are clearly supported by the papers provided.
- If the papers span fewer than 3 years, you may return fewer events or an empty array [].
- Return ONLY the JSON array. No explanation, no markdown fences.
"""


def build_paradigm_timeline(
    extractions: list[dict],
    topic: str,
) -> list[dict]:
    """
    Pass 4 (Historian): Identify paradigm shifts across the paper corpus.

    Sends all extracted paper summaries to Gemini in a single call, sorted
    chronologically, and asks it to reason about the historical arc.

    Args:
        extractions: Output of extract_paper_knowledge().
        topic:       The original user research topic (for context in the prompt).

    Returns:
        List of timeline event dicts, sorted by year ascending.
    """
    print(f"\n[Analyst] Pass 4 (Historian): Building paradigm timeline …")

    # Sort by year, put papers without a year at the end
    sorted_papers = sorted(
        extractions,
        key=lambda x: (x.get("year") is None, x.get("year") or 9999),
    )

    # Build a slim summary list for the prompt
    slim = [
        {
            "paperId":            p.get("paperId", ""),
            "title":              p.get("title", ""),
            "year":               p.get("year"),
            "methodology":        p.get("methodology", ""),
            "methodology_cluster": p.get("methodology_cluster", ""),
            "key_finding":        p.get("key_finding", ""),
        }
        for p in sorted_papers
    ]

    prompt = _TIMELINE_PROMPT.format(
        topic=topic,
        papers_json=json.dumps(slim, indent=2),
    )

    raw = _call_llm(prompt, "paradigm_timeline")
    parsed = _parse_json_safe(raw, "paradigm_timeline") if raw else None

    if parsed and isinstance(parsed, list):
        # Sort by year ascending for the Showcase's timeline chart
        timeline = sorted(parsed, key=lambda e: (e.get("year") is None, e.get("year") or 9999))
        print(f"[Analyst] Pass 4 complete — {len(timeline)} paradigm events identified.")
        return timeline

    print(f"[Analyst] Pass 4 failed — returning empty timeline.")
    return []


# ── Main entry point ──────────────────────────────────────────────────────────

def run(context: SharedContext) -> SharedContext:
    """
    Execute the full Analyst pipeline and populate SharedContext.

    This is the only function the orchestrator (main.py) calls.

    Reads:
        context.user_confirmed_papers  — post-HITL paper list
        context.paper_embeddings       — pre-computed by Retriever (do not recompute)
        context.topic                  — original user topic string

    Writes:
        context.extracted_knowledge
        context.contradiction_report
        context.paradigm_timeline
        context.pipeline_status → 'showcasing'

    Args:
        context: SharedContext with user_confirmed_papers populated.

    Returns:
        The same context object, mutated with analysis results.
    """
    context.pipeline_status = "analysing"

    papers = context.user_confirmed_papers
    if not papers:
        _log_error(
            context,
            "analyst_entry",
            "No confirmed papers in context. Did the HITL checkpoint run?",
            recoverable=False,
        )
        context.pipeline_status = "error"
        return context

    print(f"\n{'='*60}")
    print(f"[Analyst] Starting analysis on {len(papers)} confirmed papers")
    print(f"[Analyst] Topic: '{context.topic}'")
    print(f"[Analyst] Pre-computed embeddings available: {len(context.paper_embeddings)}")
    print(f"{'='*60}")

    # ── Pass 1: Extract structured knowledge ─────────────────────────────────
    try:
        extractions = extract_paper_knowledge(papers)
        context.extracted_knowledge = extractions
    except Exception as exc:
        _log_error(context, "pass1_extraction", str(exc), recoverable=False)
        context.pipeline_status = "error"
        return context

    if not extractions:
        _log_error(
            context, "pass1_extraction",
            "Extraction returned empty list — cannot proceed.",
            recoverable=False,
        )
        context.pipeline_status = "error"
        return context

    # ── Pass 2: Cluster methodologies ─────────────────────────────────────────
    try:
        clusters = cluster_methodologies(extractions, context.paper_embeddings)
    except Exception as exc:
        _log_error(context, "pass2_clustering", str(exc), recoverable=True)
        # Non-fatal: fall back to a single cluster so Pass 3 can still run
        clusters = {"all": extractions}

    # ── Pass 3: Contradiction detection ───────────────────────────────────────
    try:
        contradiction_report = detect_contradictions(clusters)
        context.contradiction_report = contradiction_report
    except Exception as exc:
        _log_error(context, "pass3_contradictions", str(exc), recoverable=True)
        context.contradiction_report = []
        # Non-fatal: pipeline continues, Showcase will note absence of report

    # ── Pass 4: Paradigm timeline ──────────────────────────────────────────────
    try:
        timeline = build_paradigm_timeline(extractions, context.topic)
        context.paradigm_timeline = timeline
    except Exception as exc:
        _log_error(context, "pass4_timeline", str(exc), recoverable=True)
        context.paradigm_timeline = []

    # ── Summary ───────────────────────────────────────────────────────────────
    major_contradictions = [
        c for c in context.contradiction_report if c.get("severity") == "major"
    ]
    low_confidence_papers = [
        e for e in context.extracted_knowledge
        if e.get("confidence_score", 1.0) < MIN_CONFIDENCE
    ]

    print(f"\n[Analyst] ✓ Analysis complete.")
    print(f"[Analyst]   Papers extracted:          {len(context.extracted_knowledge)}")
    print(f"[Analyst]   Clusters found:            {len(clusters)}")
    print(f"[Analyst]   Contradictions detected:   {len(context.contradiction_report)} "
          f"({len(major_contradictions)} major)")
    print(f"[Analyst]   Paradigm shifts identified: {len(context.paradigm_timeline)}")
    print(f"[Analyst]   Low-confidence extractions: {len(low_confidence_papers)}")

    context.pipeline_status = "showcasing"
    return context
