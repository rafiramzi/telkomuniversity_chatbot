import chromadb
import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))


_client = None
_collection = None


def embed_texts(texts):
    response = co.embed(
        model="embed-multilingual-v3.0",
        texts=texts,
        input_type="search_document"
    )
    return response.embeddings


def get_collection(cohere_ef=None):
    """
    Returns the Chroma collection.

    IMPORTANT: hnsw:space="cosine" supaya distance ada di rentang 0..1
    (0 = paling mirip, 1 = paling beda). Default Chroma adalah squared L2
    yang nilainya bisa > 1, dan itu yang bikin filter `dist < 0.6` di
    model2 selalu kosong.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.Client()
        _collection = _client.get_or_create_collection(
            name="pdf_docs_cohere",
            embedding_function=cohere_ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(docs):
    """Helper batch insert (tidak dipakai di views, tapi tetap dipertahankan)."""
    collection = get_collection()
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)
    collection.add(
        ids=[d["id"] for d in docs],
        documents=texts,
        metadatas=[{"category": d["category"]} for d in docs],
        embeddings=embeddings,
    )


def search(query, n_results=8, category=None):
    """Semantic search. Mengembalikan dict standar Chroma."""
    collection = get_collection()

    query_embedding = co.embed(
        model="embed-multilingual-v3.0",
        texts=[query],
        input_type="search_query",
    ).embeddings[0]

    where_filter = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )

    return results