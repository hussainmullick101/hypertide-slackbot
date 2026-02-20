"""Semantic search retriever against the ChromaDB knowledge base."""

from config import RAG_TOP_K
from knowledge_base.embedder import embed_query
from knowledge_base.store import query_by_embedding


def retrieve(query_text, top_k=None):
    """Embed a query and return the top-k most similar Q&A pairs.

    Returns a list of dicts with keys: document, metadata, distance.
    """
    top_k = top_k or RAG_TOP_K

    query_embedding = embed_query(query_text)
    results = query_by_embedding(query_embedding, n_results=top_k)

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "document": doc,
            "metadata": meta,
            "distance": dist,
        })

    return hits
