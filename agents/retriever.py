import datetime
import time
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI

from core.context import SharedContext
from core.embeddings import get_embedding, batch_embeddings, cosine_similarity
from tools.dedup_cache import DedupCache
from tools.semantic_scholar import search_papers, get_paper_references
from tools.arxiv_api import bulk_enrich

load_dotenv()

# ── Tuning constants ──────────────────────────────────────────────────────────

HARD_CAP = 20               # Absolute maximum papers passed to Analyst
SNOWBALL_TOP_N = 3          # How many highly-cited papers to snowball
SNOWBALL_REF_LIMIT = 15     # Max references fetched per snowball paper
SS_PAPERS_PER_QUERY = 3    # Semantic Scholar results per query variant
QUERY_VARIANTS_COUNT = 4    # Number of query variants to generate
INTER_QUERY_DELAY = 6     # Seconds between Semantic Scholar calls

class QueryOutput(BaseModel):
    queries: List[str]

_LLM = model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2,
).with_structured_output(QueryOutput)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_error(
    context: SharedContext,
    stage: str,
    message: str,
    recoverable: bool = True,
) -> None:
    """Append a structured error to context.errors and print a console warning."""
    entry = {
        "stage": stage,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "recoverable": recoverable,
    }
    context.errors.append(entry)
    level = "WARN" if recoverable else "ERROR"
    print(f"[Retriever][{stage}] {level}: {message}")


# ── Step 1: Query variant generation ─────────────────────────────────────────

def generate_query_variants(topic: str, n: int = QUERY_VARIANTS_COUNT) -> list[str]:
    global response
    prompt = f"""You are a research librarian helping build a comprehensive literature review.

                Research topic: "{topic}"

                Generate exactly {n} diverse academic search queries for Semantic Scholar and ArXiv.

                Return ONLY a valid JSON array of {n} strings. No explanation, no markdown fences.
                
                STRICT RULES:
                - Each query must be 2–6 words only
                - DO NOT use quotes ("")
                - DO NOT use AND, OR, NOT, parentheses, or any boolean operators
                - DO NOT use punctuation except spaces
                - Keep queries simple keyword-style (like a Google search)
                - Each query should represent a different angle of the topic

                Return ONLY a JSON array of strings.
                """

    try:
        response = _LLM.invoke(prompt)

    except Exception as e:
        print(f"[Retriever] Query generation failed: {e}")

    return response.queries


# ── Step 2: Semantic Scholar multi-query fetch ────────────────────────────────

def _fetch_from_semantic_scholar(query_variants: list[str]) -> list[dict]:
    """
    Run each query variant against Semantic Scholar, aggregate, and deduplicate
    by paperId (fast path — full semantic dedup happens later).
    """
    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for query in query_variants:
        print(f"[Retriever] Semantic Scholar search: '{query}'")
        try:
            results = search_papers(query, limit=SS_PAPERS_PER_QUERY)
            added = 0
            for p in results:
                pid = p.get("paperId", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)
                    added += 1
            print(f"[Retriever]   → {added} new papers (running total: {len(all_papers)})")
        except Exception as exc:
            print(f"[Retriever] SS query failed for '{query}': {exc}")

        time.sleep(INTER_QUERY_DELAY)

    return all_papers


# ── Step 3: Citation snowballing ──────────────────────────────────────────────

def _snowball_citations(base_papers: list[dict]) -> list[dict]:
    """
    Fetch the reference lists of the top-SNOWBALL_TOP_N papers by citation
    count (1 level deep).  Papers already in base_papers are skipped.
    """
    sorted_by_citations = sorted(
        base_papers,
        key=lambda p: p.get("citationCount", 0),
        reverse=True,
    )
    seed_papers = sorted_by_citations[:SNOWBALL_TOP_N]
    existing_ids: set[str] = {p["paperId"] for p in base_papers if p.get("paperId")}

    snowballed: list[dict] = []
    new_ids: set[str] = set()

    for seed in seed_papers:
        title_preview = seed.get("title", "?")[:60]
        print(f"[Retriever] Snowballing: '{title_preview}' ({seed.get('citationCount', 0)} citations)")
        try:
            refs = get_paper_references(seed["paperId"], limit=SNOWBALL_REF_LIMIT)
            added = 0
            for ref in refs:
                pid = ref.get("paperId", "")
                # Skip if already in base set or already snowballed, or no abstract
                if pid and pid not in existing_ids and pid not in new_ids and ref.get("abstract"):
                    new_ids.add(pid)
                    snowballed.append(ref)
                    added += 1
            print(f"[Retriever]   → {added} new snowball candidates")
        except Exception as exc:
            print(f"[Retriever] Snowball failed for {seed.get('paperId')}: {exc}")

        time.sleep(0.4)

    print(f"[Retriever] Snowballing added {len(snowballed)} unique candidates.")
    return snowballed


# ── Step 5: Deduplication ─────────────────────────────────────────────────────

def _deduplicate(
    papers: list[dict],
    cache: DedupCache,
) -> tuple[list[dict], dict[str, list[float]]]:
    """
    Remove semantic duplicates using ChromaDB cosine similarity.

    Returns:
        (unique_papers, paper_embeddings)
        where paper_embeddings maps paperId/arxiv_id → embedding vector.
        Storing embeddings here avoids re-computing them in the Analyst's
        clustering step.
    """
    print(f"[Retriever] Computing {len(papers)} embeddings for deduplication …")
    texts = [
        f"{p.get('title', '')} {p.get('abstract', '')}".strip() or p.get("title", "")
        for p in papers
    ]
    all_embeddings = batch_embeddings(texts)

    unique_papers: list[dict] = []
    unique_embeddings: dict[str, list[float]] = {}

    for paper, emb in zip(papers, all_embeddings):
        if not cache.is_duplicate(paper, emb):
            cache.add_paper(paper, emb)
            unique_papers.append(paper)
            pid = paper.get("paperId") or paper.get("arxiv_id", "")
            if pid:
                unique_embeddings[pid] = emb

    print(
        f"[Retriever] Dedup complete: {len(papers)} → {len(unique_papers)} unique papers."
    )
    return unique_papers, unique_embeddings


# ── Step 6: Relevance scoring ─────────────────────────────────────────────────

def _score_relevance(
    papers: list[dict],
    paper_embeddings: dict[str, list[float]],
    topic: str,
) -> list[dict]:
    """
    Compute cosine similarity between each paper's embedding and the
    original topic embedding.  Writes relevance_score into each paper dict.
    """
    print(f"[Retriever] Scoring relevance against topic: '{topic}'")
    topic_embedding = get_embedding(topic)

    for paper in papers:
        pid = paper.get("paperId") or paper.get("arxiv_id", "")
        emb = paper_embeddings.get(pid)
        if emb:
            paper["relevance_score"] = cosine_similarity(topic_embedding, emb)
        else:
            # Fallback: compute embedding on-the-fly (shouldn't happen normally)
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
            if text:
                paper_emb = get_embedding(text)
                paper["relevance_score"] = cosine_similarity(topic_embedding, paper_emb)
                paper_embeddings[pid] = paper_emb
            else:
                paper["relevance_score"] = 0.0

    return sorted(papers, key=lambda p: p.get("relevance_score", 0.0), reverse=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(context: SharedContext) -> SharedContext:
    """
    Execute the full Retriever pipeline and populate SharedContext.

    This is the only function the orchestrator (main.py) calls.

    Args:
        context: SharedContext with context.topic set.

    Returns:
        The same context object, mutated with retrieval results.
    """
    context.pipeline_status = "retrieving"
    cache = DedupCache()
    cache.reset()  # Always start fresh — no cross-topic contamination

    # ── 1. Query variant generation ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Retriever] Starting retrieval for topic: '{context.topic}'")
    print(f"{'='*60}")

    try:
        context.query_variants = generate_query_variants(context.topic)
        print(f"[Retriever] Query variants: {context.query_variants}")
    except Exception as exc:
        _log_error(context, "query_generation", str(exc), recoverable=True)
        context.query_variants = [context.topic]

    # ── 2. Semantic Scholar fetch ─────────────────────────────────────────────
    try:
        ss_papers = _fetch_from_semantic_scholar(context.query_variants)
        print(f"[Retriever] Semantic Scholar total: {len(ss_papers)} papers")
    except Exception as exc:
        _log_error(context, "semantic_scholar_fetch", str(exc), recoverable=False)
        context.pipeline_status = "error"
        return context

    if not ss_papers:
        _log_error(
            context,
            "semantic_scholar_fetch",
            f"Zero papers returned for topic '{context.topic}'. "
            "Try a broader or differently-phrased topic.",
            recoverable=False,
        )
        context.pipeline_status = "error"
        return context

    # ── 3. Citation snowballing ───────────────────────────────────────────────
    snowballed: list[dict] = []
    try:
        snowballed = _snowball_citations(ss_papers)
    except Exception as exc:
        _log_error(context, "snowballing", str(exc), recoverable=True)
        # Non-fatal — we proceed without snowball results

    all_candidates = ss_papers + snowballed
    context.raw_papers = list(all_candidates)  # Store pre-dedup snapshot
    print(f"[Retriever] Total candidates before dedup: {len(all_candidates)}")

    # ── 4. ArXiv enrichment ───────────────────────────────────────────────────
    # ── 4. ArXiv enrichment (LIMITED) ──────────────────────────────────────────
    try:
        ENRICH_TOP_K = 20  # 🔥 limit API calls here

        # Sort by citation count so we enrich the most important papers first
        all_candidates = sorted(
            all_candidates,
            key=lambda p: p.get("citationCount", 0),
            reverse=True
        )

        top_for_enrichment = all_candidates[:ENRICH_TOP_K]
        remaining = all_candidates[ENRICH_TOP_K:]

        print(f"[Retriever] Enriching top {ENRICH_TOP_K} papers with ArXiv metadata...")

        enriched_top = bulk_enrich(top_for_enrichment)

        # Merge back
        all_candidates = enriched_top + remaining

    except Exception as exc:
        _log_error(context, "arxiv_enrichment", str(exc), recoverable=True)

    # ── 5. Deduplication ──────────────────────────────────────────────────────
    try:
        unique_papers, paper_embeddings = _deduplicate(all_candidates, cache)
    except Exception as exc:
        _log_error(context, "deduplication", str(exc), recoverable=False)
        context.pipeline_status = "error"
        return context

    # ── 6. Relevance scoring ──────────────────────────────────────────────────
    try:
        unique_papers = _score_relevance(unique_papers, paper_embeddings, context.topic)
    except Exception as exc:
        _log_error(context, "relevance_scoring", str(exc), recoverable=True)
        # Non-fatal — papers will have default relevance_score=0.0

    # ── 7. Hard cap ───────────────────────────────────────────────────────────
    if len(unique_papers) > HARD_CAP:
        print(f"[Retriever] Enforcing hard cap: {len(unique_papers)} → {HARD_CAP} papers")
        unique_papers = unique_papers[:HARD_CAP]
        # Trim the embeddings dict to only kept papers
        kept_ids = {
            p.get("paperId") or p.get("arxiv_id", "")
            for p in unique_papers
        }
        paper_embeddings = {k: v for k, v in paper_embeddings.items() if k in kept_ids}

    # ── Write results to context ──────────────────────────────────────────────
    context.deduplicated_papers = unique_papers
    context.paper_embeddings = paper_embeddings

    print(f"\n[Retriever] ✓ Done — {len(unique_papers)} papers ready for HITL review.")
    print(
        f"[Retriever]   Top 3 by relevance: "
        + ", ".join(
            f"'{p.get('title', '?')[:40]}' ({p.get('relevance_score', 0):.2f})"
            for p in unique_papers[:3]
        )
    )

    context.pipeline_status = "awaiting_hitl"
    return context