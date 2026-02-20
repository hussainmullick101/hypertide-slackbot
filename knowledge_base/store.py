"""ChromaDB operations for the support email knowledge base."""

import chromadb

from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME


_client = None
_collection = None


def get_client():
    """Get or create a persistent ChromaDB client."""
    global _client
    if _client is None:
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _client


def get_collection():
    """Get or create the support emails collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "Hypertide support email Q&A pairs"},
        )
    return _collection


def add_documents(ids, embeddings, documents, metadatas):
    """Add documents with embeddings and metadata to the collection."""
    collection = get_collection()

    # ChromaDB has a batch size limit, process in chunks
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    return len(ids)


def query_by_embedding(query_embedding, n_results=5):
    """Query the collection by embedding similarity."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results


def get_count():
    """Return the number of documents in the collection."""
    return get_collection().count()


def reset_collection():
    """Delete and recreate the collection."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except ValueError:
        pass
    _collection = None
    return get_collection()
