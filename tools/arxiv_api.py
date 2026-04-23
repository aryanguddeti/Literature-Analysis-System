"""
tools/arxiv_api.py
──────────────────
ArXiv API wrapper using the official `arxiv` Python library.
Owned by Aryan (Retriever).

Primary job:  given a paper already found via Semantic Scholar, locate the
              same paper on ArXiv so we can attach the PDF URL and fill any
              missing abstracts.

Strategy per paper:
  1. If the Semantic Scholar record already has an arxiv_id → fetch directly.
  2. Otherwise search ArXiv by exact title (quoted), fall back to unquoted.
  3. If still nothing → leave pdf_url as None (paper may be non-ArXiv).

We do NOT download full PDFs here — we only collect metadata + pdf_url.
PDF text extraction (PyMuPDF) is a Phase 2 enhancement owned by Anmol's Analyst.
"""

import time
import random
from typing import Optional

import arxiv

# ── ArXiv client configuration ────────────────────────────────────────────────
# delay_seconds=1.0 keeps us well within ArXiv's access guidelines.
_CLIENT = arxiv.Client(
    page_size=5,
    delay_seconds=3.0,
    num_retries=3,
)


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize_result(result: arxiv.Result) -> dict:
    """
    Convert an arxiv.Result into the SharedContext paper dict schema.

    Note: citationCount and references are always 0 / [] here —
    those are only populated by Semantic Scholar.
    """
    # Strip version suffix: "2017.12345v3" → "2017.12345"
    raw_id: str = result.entry_id.split("/")[-1]
    arxiv_id: str = raw_id.split("v")[0]

    return {
        "paperId": f"arxiv:{arxiv_id}",
        "arxiv_id": arxiv_id,
        "title": result.title.strip(),
        "abstract": result.summary.replace("\n", " ").strip(),
        "year": result.published.year if result.published else None,
        "authors": [str(a) for a in result.authors],
        "citationCount": 0,
        "references": [],
        "source": "arxiv",
        "pdf_url": result.pdf_url,
        "relevance_score": 0.0,
    }


# ── Search helpers ────────────────────────────────────────────────────────────

def search_arxiv_by_title(title: str, max_results: int = 3) -> Optional[dict]:
    """
    Search ArXiv for *title* and return the best matching paper, or None.

    Two-pass strategy:
      Pass 1 — exact quoted title search: ti:"<title>"
      Pass 2 — unquoted broader search:   ti:<title>
    """
    if not title:
        return None

    # Sanitise title for query: remove special chars that break the ArXiv query parser
    safe_title = title.replace('"', '').replace("'", "").strip()

    for query in [f'ti:"{safe_title}"', f"ti:{safe_title}"]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            results = list(_CLIENT.results(search))
            if results:
                # Pick the result whose title most closely matches the query title
                best = _pick_best_title_match(results, title)
                if best:
                    return _normalize_result(best)
        except Exception as exc:
            print(f"[ArXiv] search_by_title failed for '{title[:60]}': {exc}")

    return None


def fetch_by_arxiv_id(arxiv_id: str) -> Optional[dict]:
    """
    Fetch a specific paper directly by its ArXiv ID (e.g. '1706.03762').
    Strips version suffix automatically.

    Returns:
        Normalised paper dict, or None if not found.
    """
    clean_id = arxiv_id.split("v")[0].strip()
    search = arxiv.Search(id_list=[clean_id])
    try:
        results = list(_CLIENT.results(search))
        if results:
            return _normalize_result(results[0])
    except Exception as exc:
        print(f"[ArXiv] fetch_by_id failed for '{arxiv_id}': {exc}")
    return None


# ── Title matching helper ─────────────────────────────────────────────────────

def _pick_best_title_match(
    results: list[arxiv.Result],
    target_title: str,
) -> Optional[arxiv.Result]:
    """
    From a list of ArXiv results, return the one whose title is most similar
    to *target_title* using a simple normalised token overlap score.

    Falls back to the first result if overlap can't determine a winner.
    """
    def _tokens(s: str) -> set[str]:
        return set(s.lower().split())

    target_tokens = _tokens(target_title)
    if not target_tokens:
        return results[0] if results else None

    best_result = results[0]
    best_score = -1.0

    for result in results:
        result_tokens = _tokens(result.title)
        if not result_tokens:
            continue
        overlap = len(target_tokens & result_tokens)
        union = len(target_tokens | result_tokens)
        score = overlap / union if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_result = result

    # If the best match shares fewer than 40% of tokens, it's probably wrong
    return best_result if best_score >= 0.4 else None


# ── Enrichment (main public function called by Retriever) ─────────────────────

def enrich_paper_with_arxiv(paper: dict) -> dict:
    """
    Given a Semantic Scholar paper dict, attach ArXiv-sourced fields:
      - pdf_url     (always populated if found on ArXiv)
      - arxiv_id    (confirmed or newly discovered)
      - abstract    (filled only if Semantic Scholar returned empty string)

    The original dict is NOT mutated — a shallow copy is returned.

    Strategy:
      1. If paper already has arxiv_id → fetch directly (fast, reliable).
      2. Otherwise → search by title (slower, may miss paywalled or old papers).

    Args:
        paper: A normalised paper dict from semantic_scholar.py.

    Returns:
        Updated paper dict with ArXiv fields populated where available.
    """
    paper = dict(paper)  # shallow copy — never mutate shared context data in-place

    arxiv_data: Optional[dict] = None

    # ── Path 1: direct fetch by known ArXiv ID ────────────────────────────────
    if paper.get("arxiv_id"):
        arxiv_data = fetch_by_arxiv_id(paper["arxiv_id"])

    # ── Path 2: title search fallback ─────────────────────────────────────────
    if arxiv_data is None and paper.get("title"):
        arxiv_data = search_arxiv_by_title(paper["title"])

    # ── Merge ArXiv fields into the paper dict ────────────────────────────────
    if arxiv_data:
        paper["pdf_url"] = arxiv_data.get("pdf_url")

        # Confirm or set arxiv_id
        if arxiv_data.get("arxiv_id"):
            paper["arxiv_id"] = arxiv_data["arxiv_id"]

        # Fill abstract only if Semantic Scholar returned nothing
        if not paper.get("abstract") and arxiv_data.get("abstract"):
            paper["abstract"] = arxiv_data["abstract"]

    return paper


def bulk_enrich(papers: list[dict], delay: float = 2.0) -> list[dict]:
    """
    Run enrich_paper_with_arxiv on a list of papers with a small inter-call
    delay to respect ArXiv's access guidelines.

    Args:
        papers: List of Semantic Scholar normalised paper dicts.
        delay:  Seconds to wait between ArXiv API calls.

    Returns:
        New list of enriched paper dicts (originals untouched).
    """
    enriched: list[dict] = []
    total = len(papers)

    for i, paper in enumerate(papers):
        title_preview = paper.get("title", "?")[:55]
        print(f"[ArXiv] Enriching {i + 1}/{total}: '{title_preview}'")
        enriched.append(enrich_paper_with_arxiv(paper))
        if i < total - 1:
            time.sleep(delay + random.uniform(0.2, 0.8))

    arxiv_found = sum(1 for p in enriched if p.get("pdf_url"))
    print(f"[ArXiv] Enrichment complete. {arxiv_found}/{total} papers have PDF URLs.")
    return enriched