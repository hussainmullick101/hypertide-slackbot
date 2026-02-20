"""Sentence-transformers embedding wrapper."""

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


_model = None


def get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    """Embed a list of texts, returning a list of embedding vectors."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 100)
    return embeddings.tolist()


def embed_query(query):
    """Embed a single query string."""
    model = get_model()
    return model.encode(query).tolist()


def build_document_text(qa_pair):
    """Combine Q&A pair fields into a single document string for embedding."""
    parts = []
    if qa_pair.get("subject"):
        parts.append(f"Subject: {qa_pair['subject']}")
    if qa_pair.get("question"):
        parts.append(f"Customer question: {qa_pair['question']}")
    if qa_pair.get("response"):
        parts.append(f"Support response: {qa_pair['response']}")
    return "\n\n".join(parts)
