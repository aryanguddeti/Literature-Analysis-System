# import requests
#
# BASE_URL = "https://api.semanticscholar.org/graph/v1"
#
# def search_papers(query: str, limit: int = 5) -> list[dict]:
#     params = {
#         "query": query,
#         "limit": limit,
#         "fields": "paperId,title,abstract,year,authors,citationCount,references,externalIds"
#     }
#     r = requests.get(f"{BASE_URL}/paper/search", params=params)
#     return r.json().get("data", [])
#
# def get_paper_references(paper_id: str) -> list[dict]:
#     """Used for citation snowballing — fetch 1 level of references."""
#     r = requests.get(
#         f"{BASE_URL}/paper/{paper_id}/references",
#         params={"fields": "paperId,title,abstract,year,citationCount"}
#     )
#     return r.json().get("data", [])





"""
tools/semantic_scholar.py
─────────────────────────
Semantic Scholar Academic Graph API wrapper.
Owned by Aryan (Retriever).

Endpoints used:
  GET /paper/search          — keyword search
  GET /paper/{id}/references — citation snowballing (papers this paper cites)
  GET /paper/{id}            — single-paper detail fetch

All responses are normalised to the SharedContext paper dict schema before
being returned, so the rest of the pipeline never touches raw API payloads.

Rate limiting: free tier allows ~1 req/s; we sleep conservatively between calls.
No API key required for basic access. Set SEMANTIC_SCHOLAR_API_KEY in .env for
higher rate limits (optional).
"""

import os
import time
import requests
from typing import Optional

# ── API Configuration ─────────────────────────────────────────────────────────

_BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields to request for search and detail endpoints
_PAPER_FIELDS = (
    "paperId,title,abstract,year,authors,citationCount,"
    "references,externalIds,publicationDate"
)
# Lighter field set for reference/snowball calls (references-of-references not needed)
_REF_FIELDS = (
    "paperId,title,abstract,year,authors,citationCount,externalIds"
)

_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": "LiteratureAnalysisSystem/1.0 (academic research project)",
}


def _get_headers() -> dict[str, str]:
    headers = dict(_DEFAULT_HEADERS)
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(
    url: str,
    params: dict,
    retries: int = 3,
    base_delay: float = 2.0,
) -> Optional[dict]:
    """
    GET *url* with exponential back-off on 429 / transient errors.

    Returns parsed JSON dict or None on permanent failure.
    """
    headers = _get_headers()
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)

            if resp.status_code == 429:
                # Rate limited — back off and retry
                wait = base_delay * (2 ** attempt)
                print(f"[SemanticScholar] Rate limited. Waiting {wait:.1f}s …")
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                return None  # Paper not found — not a retryable error

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            print(f"[SemanticScholar] Timeout (attempt {attempt + 1}/{retries})")
        except requests.exceptions.RequestException as exc:
            print(f"[SemanticScholar] Request error: {exc} (attempt {attempt + 1}/{retries})")

        if attempt < retries - 1:
            time.sleep(base_delay * (attempt + 1))

    return None


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize_paper(raw: dict, source: str = "semantic_scholar") -> dict:
    """
    Convert a raw Semantic Scholar paper object into the SharedContext schema.

    SharedContext paper dict keys:
        paperId         str   — Semantic Scholar internal ID
        title           str
        abstract        str   — empty string if not available
        year            int | None
        authors         list[str]
        citationCount   int
        references      list[str]  — list of cited paperIds
        source          str   — 'semantic_scholar' | 'snowball' | 'arxiv'
        arxiv_id        str | None — ArXiv ID if present in externalIds
        pdf_url         str | None — filled later by ArXiv enrichment
        relevance_score float — filled later by Retriever scoring step
    """
    authors: list[str] = [
        a.get("name", "") for a in (raw.get("authors") or []) if a.get("name")
    ]

    # Extract cited paper IDs from the references list
    references: list[str] = []
    for ref in raw.get("references") or []:
        if isinstance(ref, dict) and ref.get("paperId"):
            references.append(ref["paperId"])

    # Dig out ArXiv ID from externalIds if present
    ext_ids: dict = raw.get("externalIds") or {}
    arxiv_id: Optional[str] = ext_ids.get("ArXiv") or ext_ids.get("arxiv")

    return {
        "paperId": raw.get("paperId", ""),
        "title": (raw.get("title") or "").strip(),
        "abstract": (raw.get("abstract") or "").strip(),
        "year": raw.get("year"),
        "authors": authors,
        "citationCount": int(raw.get("citationCount") or 0),
        "references": references,
        "source": source,
        "arxiv_id": arxiv_id,
        "pdf_url": None,           # Populated by arxiv_api.enrich_paper_with_arxiv
        "relevance_score": 0.0,    # Populated by Retriever._score_relevance
    }


# ── Public API ────────────────────────────────────────────────────────────────

def search_papers(
    query: str,
    limit: int = 20,
    offset: int = 0,
    min_citation_count: int = 0,
) -> list[dict]:
    """
    Search Semantic Scholar for papers matching *query*.

    Args:
        query:              Free-text search string.
        limit:              Max results to return (API cap: 100).
        offset:             Pagination offset.
        min_citation_count: Filter out papers with fewer citations than this.
                            Useful for surfacing established work.

    Returns:
        List of normalised paper dicts.
    """
    params: dict = {
        "query": query,
        "limit": min(limit, 100),
        "offset": offset,
        "fields": _PAPER_FIELDS,
    }

    data = _get(f"{_BASE_URL}/paper/search", params)
    if not data or "data" not in data:
        return []

    papers = []
    for raw in data["data"]:
        if not raw.get("paperId"):
            continue
        paper = _normalize_paper(raw, source="semantic_scholar")
        if paper["citationCount"] >= min_citation_count:
            papers.append(paper)

    print(
        f"[SemanticScholar] '{query[:60]}' → {len(papers)} papers "
        f"(total available: {data.get('total', '?')})"
    )
    time.sleep(5)  # Polite rate limiting on free tier
    return papers


def get_paper_references(
    paper_id: str,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch papers *cited by* paper_id — used for citation snowballing.

    We request the papers that this paper cites (its reference list),
    not the papers that cite it (which would be /citations and can be huge).

    Args:
        paper_id: Semantic Scholar paper ID.
        limit:    Max references to fetch (API cap: 1000).

    Returns:
        List of normalised paper dicts with source='snowball'.
    """
    params: dict = {
        "fields": _REF_FIELDS,
        "limit": min(limit, 1000),
    }

    data = _get(f"{_BASE_URL}/paper/{paper_id}/references", params)
    if not data or "data" not in data:
        return []

    papers = []
    for item in data["data"]:
        cited = item.get("citedPaper") or {}
        if cited.get("paperId") and cited.get("title"):
            papers.append(_normalize_paper(cited, source="snowball"))

    time.sleep(1.4)
    return papers


def get_paper_details(paper_id: str) -> Optional[dict]:
    """
    Fetch full metadata for a single paper by its Semantic Scholar ID.

    Returns:
        Normalised paper dict, or None if not found.
    """
    params: dict = {"fields": _PAPER_FIELDS}
    data = _get(f"{_BASE_URL}/paper/{paper_id}", params)
    if not data or not data.get("paperId"):
        return None
    return _normalize_paper(data)


def search_papers_paginated(
    query: str,
    total: int = 60,
    page_size: int = 20,
) -> list[dict]:
    """
    Convenience wrapper that paginates through results to retrieve *total* papers.
    Useful when a single query needs more than the default 20 results.

    Args:
        query:     Search query string.
        total:     Total papers to fetch (hard cap applied by Retriever anyway).
        page_size: Results per API call.

    Returns:
        Flat list of normalised paper dicts.
    """
    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for offset in range(0, total, page_size):
        batch = search_papers(query, limit=page_size, offset=offset)
        if not batch:
            break
        for p in batch:
            if p["paperId"] not in seen_ids:
                seen_ids.add(p["paperId"])
                all_papers.append(p)

    return all_papers