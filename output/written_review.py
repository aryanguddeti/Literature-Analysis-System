"""
output/written_review.py
────────────────────────
Format 1: Written Synthesis Review — Gemini-powered structured markdown essay.
Owned by Devang.

Produces a full literature review essay covering:
  - Introduction & research landscape
  - Thematic analysis by methodology cluster
  - Contradictions & open debates
  - Historical trajectory (from paradigm timeline)
  - Future directions & research gaps
  - Conclusion

The review reads like a human-written survey paper, not a list of summaries.
"""

import json
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from core.context import SharedContext

_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=8192,
    timeout=120,
    max_retries=3,
)

_REVIEW_PROMPT = """You are a senior academic researcher writing a comprehensive literature review.

Topic: "{topic}"

You have analyzed {n_papers} papers. Here is the structured knowledge extracted:

=== PAPERS BY CLUSTER ===
{cluster_summaries}

=== CONTRADICTIONS FOUND ===
{contradictions_text}

=== PARADIGM TIMELINE ===
{timeline_text}

Write a comprehensive literature review essay in markdown format with these sections:

## 1. Introduction
Brief overview of the research area, its importance, and scope of this review.

## 2. Research Landscape
Overview of the main methodological approaches and how many papers fall into each cluster.

## 3. Thematic Analysis
A section for EACH methodology cluster (use the cluster names as subsections).
For each cluster: discuss the core ideas, key papers, how they relate to each other, and their strengths/weaknesses.

## 4. Contradictions and Open Debates  
Discuss the contradictions found. For major contradictions, explain both sides.
If no contradictions found, discuss the areas of consensus.

## 5. Historical Trajectory
Narrative of how this field evolved over time, using the paradigm timeline data.

## 6. Research Gaps and Future Directions
What questions remain unanswered? What approaches haven't been tried?

## 7. Conclusion
2-3 paragraph synthesis of the state of the art.

REQUIREMENTS:
- Write in flowing academic prose — NOT bullet points
- Cite papers by their title and year, e.g. (Vaswani et al., 2017)
- Be specific about methods and results, not vague
- Total length: 800-1200 words
- Return ONLY the markdown essay, no preamble
"""


def _build_cluster_summaries(context: SharedContext) -> str:
    """Build a structured summary of papers grouped by cluster."""
    extractions = context.extracted_knowledge
    papers_map = {
        (p.get("paperId") or p.get("arxiv_id", "")): p
        for p in context.user_confirmed_papers
    }

    clusters: dict[str, list[dict]] = {}
    for ext in extractions:
        cluster = ext.get("methodology_cluster", "other")
        clusters.setdefault(cluster, []).append(ext)

    sections = []
    for cluster_name, exts in clusters.items():
        section = [f"Cluster: {cluster_name} ({len(exts)} papers)"]
        for ext in exts:
            pid = ext.get("paperId", "")
            paper = papers_map.get(pid, {})
            authors = paper.get("authors", [])
            author_str = authors[0].split()[-1] if authors else "Unknown"
            section.append(
                f"  - {ext.get('title', '?')} ({ext.get('year', 'N/A')}, {author_str} et al.): "
                f"{ext.get('key_finding', 'N/A')} | Method: {ext.get('methodology', 'N/A')}"
            )
        sections.append("\n".join(section))

    return "\n\n".join(sections)


def _build_contradictions_text(context: SharedContext) -> str:
    """Format contradiction report for the prompt."""
    if not context.contradiction_report:
        return "No contradictions detected in this paper set."

    lines = []
    for c in context.contradiction_report:
        severity = c.get("severity", "minor").upper()
        lines.append(
            f"[{severity}] {c.get('paper_a_title', '?')} vs {c.get('paper_b_title', '?')}: "
            f"{c.get('reasoning', '—')}"
        )
    return "\n".join(lines)


def _build_timeline_text(context: SharedContext) -> str:
    """Format paradigm timeline for the prompt."""
    if not context.paradigm_timeline:
        return "No paradigm timeline available."

    lines = []
    for event in context.paradigm_timeline:
        lines.append(
            f"{event.get('year', '?')}: {event.get('shift_description', '—')} "
            f"[{event.get('impact', 'incremental')}] — Key paper: {event.get('key_paper', '—')}"
        )
    return "\n".join(lines)


def generate(context: SharedContext) -> str:
    """
    Generate a full literature review essay in Markdown.

    Args:
        context: SharedContext with all analyst outputs populated.

    Returns:
        Markdown string of the review essay.
    """
    if not context.extracted_knowledge:
        return f"# Literature Review: {context.topic}\n\n*No analysis data available.*"

    cluster_summaries = _build_cluster_summaries(context)
    contradictions_text = _build_contradictions_text(context)
    timeline_text = _build_timeline_text(context)

    prompt = _REVIEW_PROMPT.format(
        topic=context.topic,
        n_papers=len(context.user_confirmed_papers),
        cluster_summaries=cluster_summaries,
        contradictions_text=contradictions_text,
        timeline_text=timeline_text,
    )

    try:
        response = _LLM.invoke([HumanMessage(content=prompt)])
        review_text = response.content.strip()
    except Exception as exc:
        print(f"[WrittenReview] LLM call failed: {exc}")
        review_text = f"# Literature Review: {context.topic}\n\n*Review generation failed. Please try again.*"

    # Prepend a header with metadata
    header = (
        f"# Literature Review: {context.topic}\n\n"
        f"*{len(context.user_confirmed_papers)} papers · "
        f"{len(context.extracted_knowledge)} analyzed · "
        f"{len(context.contradiction_report)} contradictions · "
        f"{len(context.paradigm_timeline)} paradigm shifts*\n\n---\n\n"
    )

    return header + review_text
