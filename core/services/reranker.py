import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def rerank(query, documents, top_n=4):
    docs_for_rerank = [{"text": d} for d in documents]

    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=docs_for_rerank,
        top_n=top_n
    )

    return [
        r.document["text"]
        for r in response.results
        if r.document and "text" in r.document
    ]