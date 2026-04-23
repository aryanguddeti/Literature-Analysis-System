"""
tools/dedup_cache.py
────────────────────
ChromaDB-backed deduplication cache.
Owned by Aryan (Retriever).

Two-layer deduplication strategy:
  1. Fast path  — exact paperId / arxiv_id string match (O(1) set lookup).
  2. Semantic   — cosine similarity of title+abstract embedding via ChromaDB
                  HNSW index. Papers above SIMILARITY_THRESHOLD are duplicates.

ChromaDB persists vectors to disk so re-runs within a session are free
(no re-embedding). Call cache.reset() at the start of each new topic run.
"""

import os
import chromadb
from chromadb.config import Settings

from core.embeddings import get_embedding, batch_embeddings

# Papers with cosine similarity ≥ this threshold are considered the same paper.
# 0.92 is tight enough to catch title variations ("Attention is All You Need" vs
# "Attention Is All You Need") while not merging distinct papers in the same cluster.
SIMILARITY_THRESHOLD = 0.92

_DEFAULT_PERSIST_DIR = ".chromadb_cache"
_COLLECTION_NAME = "papers"


class DedupCache:
    """
    Wraps a ChromaDB collection to deduplicate papers by semantic similarity.

    Usage:
        cache = DedupCache()
        cache.reset()           # always reset at the start of a new topic run

        emb = get_embedding(text)
        if not cache.is_duplicate(paper, emb):
            cache.add_paper(paper, emb)
    """

    def __init__(self, persist_dir: str = _DEFAULT_PERSIST_DIR) -> None:
        self._persist_dir = persist_dir
        self._chroma = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._get_or_create_collection()
        # Fast-path: in-memory set of IDs already in the collection.
        self._seen_ids: set[str] = set()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self._chroma.get_or_create_collection(
            name=_COLLECTION_NAME,
            # ChromaDB cosine distance = 1 − cosine_similarity
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _paper_id(paper: dict) -> str:
        """Return a stable, non-empty ID string for a paper dict."""
        return (
            paper.get("paperId")
            or paper.get("arxiv_id")
            or paper.get("title", "")[:80]
        )

    @staticmethod
    def _doc_text(paper: dict) -> str:
        """The text we embed for a paper (title + abstract)."""
        return f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_duplicate(
        self,
        paper: dict,
        embedding: list[float] | None = None,
    ) -> bool:
        """
        Return True if *paper* is already in the cache.

        Args:
            paper:     Paper dict (must have at least 'title').
            embedding: Pre-computed embedding. Computed on demand if None.
        """
        pid = self._paper_id(paper)

        # 1. Fast path: exact ID match
        if pid and pid in self._seen_ids:
            return True

        # 2. Semantic similarity check
        if embedding is None:
            embedding = get_embedding(self._doc_text(paper))

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances"],
            )
            distances = results.get("distances", [[]])
            if distances and distances[0]:
                # ChromaDB cosine distance = 1 − cosine_similarity
                similarity = 1.0 - distances[0][0]
                if similarity >= SIMILARITY_THRESHOLD:
                    return True
        except Exception:
            # Collection is empty — first query raises; not a real error.
            pass

        return False

    def add_paper(
        self,
        paper: dict,
        embedding: list[float] | None = None,
    ) -> None:
        """
        Add a single paper to the cache.

        Args:
            paper:     Paper dict.
            embedding: Pre-computed embedding. Computed on demand if None.
        """
        pid = self._paper_id(paper)
        if not pid:
            return  # Can't track a paper without any identifier

        if embedding is None:
            embedding = get_embedding(self._doc_text(paper))

        try:
            self._collection.add(
                ids=[pid],
                embeddings=[embedding],
                metadatas=[
                    {
                        "title": paper.get("title", "")[:500],
                        "source": paper.get("source", ""),
                        "year": str(paper.get("year") or ""),
                    }
                ],
                documents=[self._doc_text(paper)[:2000]],
            )
            self._seen_ids.add(pid)
        except Exception as e:
            # ChromaDB raises if the ID already exists; treat as a no-op.
            if "already exists" not in str(e).lower():
                raise

    def add_papers_batch(
        self,
        papers: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add multiple papers at once. More efficient than repeated add_paper calls
        for large initial inserts.

        Args:
            papers:     List of paper dicts.
            embeddings: Pre-computed embeddings in the same order.
        """
        if len(papers) != len(embeddings):
            raise ValueError("papers and embeddings must have the same length")

        # Filter out papers without a usable ID and deduplicate within the batch.
        seen_in_batch: set[str] = set()
        ids, embs, metas, docs = [], [], [], []

        for paper, emb in zip(papers, embeddings):
            pid = self._paper_id(paper)
            if not pid or pid in self._seen_ids or pid in seen_in_batch:
                continue
            seen_in_batch.add(pid)
            ids.append(pid)
            embs.append(emb)
            metas.append(
                {
                    "title": paper.get("title", "")[:500],
                    "source": paper.get("source", ""),
                    "year": str(paper.get("year") or ""),
                }
            )
            docs.append(self._doc_text(paper)[:2000])

        if not ids:
            return

        try:
            self._collection.add(
                ids=ids,
                embeddings=embs,
                metadatas=metas,
                documents=docs,
            )
            self._seen_ids.update(ids)
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

    def reset(self) -> None:
        """
        Wipe the collection and in-memory ID set.
        MUST be called at the start of every new topic run to avoid
        cross-topic contamination.
        """
        try:
            self._chroma.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._get_or_create_collection()
        self._seen_ids.clear()

    def __len__(self) -> int:
        return self._collection.count()