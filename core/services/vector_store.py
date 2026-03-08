import chromadb
import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))

client = chromadb.Client()
collection = client.get_or_create_collection(name="telkom_university")


def embed_texts(texts):
    response = co.embed(
        model="embed-multilingual-v3.0",
        texts=texts,
        input_type="search_document"
    )
    return response.embeddings


def add_documents(docs):
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[d["id"] for d in docs],
        documents=texts,
        metadatas=[{"category": d["category"]} for d in docs],
        embeddings=embeddings
    )


def search(query, n_results=8, category=None):
    query_embedding = co.embed(
        model="embed-multilingual-v3.0",
        texts=[query],
        input_type="search_query"
    ).embeddings[0]

    where_filter = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    return results