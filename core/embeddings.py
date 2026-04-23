"""
core/embeddings.py
──────────────────
Thin wrapper around Gemini embeddings via LangChain.

Responsibilities:
  - Single and batch embedding generation
  - Cosine similarity helper
  - Lazy client initialisation
"""

import os
import numpy as np
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None


def _get_client() -> GoogleGenerativeAIEmbeddings:
    """Lazily initialise Gemini embeddings via LangChain."""
    global _embeddings
    if _embeddings is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not found. Add it to your .env file."
            )

        _embeddings = GoogleGenerativeAIEmbeddings(
            model=MODEL,
            google_api_key=api_key,
        )

    return _embeddings


# ── Public API ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Compute a single embedding vector."""
    text = text.strip().replace("\n", " ")
    if not text:
        return [0.0] * EMBEDDING_DIM

    return _get_client().embed_query(text)


# def batch_embeddings(texts: list[str], batch_size: int = 100) -> list[list[float]]:
#     """Compute embeddings for a list of texts."""
#     all_embeddings: list[list[float]] = []
#
#     for i in range(0, len(texts), batch_size):
#         batch = [
#             t.strip().replace("\n", " ") or " "
#             for t in texts[i: i + batch_size]
#         ]
#
#         embeddings = _get_client().embed_documents(batch)
#         all_embeddings.extend(embeddings)
#
#     return all_embeddings

def batch_embeddings(texts: list[str], batch_size: int = 500) -> list[list[float]]:
    """
    Optimized batch embedding:
    - larger batch size
    - single API call per batch
    - no per-item overhead
    """

    client = _get_client()
    all_embeddings: list[list[float]] = []

    # clean once
    cleaned = [t.strip().replace("\n", " ") or " " for t in texts]

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i + batch_size]

        embeddings = client.embed_documents(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)

    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / denom)