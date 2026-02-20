"""Semantic search retriever against the ChromaDB knowledge base."""

import time
from email.utils import parsedate_tz, mktime_tz

from config import RAG_TOP_K
from knowledge_base.embedder import embed_query
from knowledge_base.store import query_by_embedding

# How much to favor recent emails (0 = pure similarity, 1 = heavy recency bias)
RECENCY_WEIGHT = 0.3


def _parse_epoch(date_str):
    """Parse an email date header into a unix timestamp, or return 0."""
    if not date_str:
        return 0
    try:
        t = parsedate_tz(date_str)
        return mktime_tz(t) if t else 0
    except Exception:
        return 0


def retrieve(query_text, top_k=None):
    """Embed a query and return the top-k results, re-ranked to favor newer emails.

    Fetches 3x candidates, scores them with a blend of similarity + recency,
    then returns the top-k.
    """
    top_k = top_k or RAG_TOP_K
    fetch_k = top_k * 3

    query_embedding = embed_query(query_text)
    results = query_by_embedding(query_embedding, n_results=fetch_k)

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    hits = []
    now = time.time()

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        epoch = _parse_epoch(meta.get("date", ""))
        # Recency score: 1.0 for today, decaying toward 0 over ~2 years
        days_ago = max((now - epoch) / 86400, 0) if epoch else 365
        recency_score = max(1.0 - (days_ago / 730), 0)

        # ChromaDB distance is L2 — lower is better. Normalize to 0-1 similarity.
        similarity = 1.0 / (1.0 + dist)

        # Blended score: higher is better
        score = (1 - RECENCY_WEIGHT) * similarity + RECENCY_WEIGHT * recency_score

        hits.append({
            "document": doc,
            "metadata": meta,
            "distance": dist,
            "score": score,
        })

    # Sort by blended score (highest first) and return top_k
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]
