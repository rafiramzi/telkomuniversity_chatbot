import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def rerank(query, documents, top_n=4):
    if not documents:
        return []

    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documents,   # ✅ kirim list of string langsung, bukan dict
        top_n=min(top_n, len(documents)),
        return_documents=True  # ✅ paksa API kembalikan teks dokumen
    )

    print(f"[RERANK] results count: {len(response.results)}")
    for r in response.results:
        print(f"  index={r.index} | score={r.relevance_score:.4f} | doc={str(r.document)[:60]}")

    # ✅ fallback: kalau r.document tetap None, ambil dari index original
    results = []
    for r in response.results:
        if r.document and hasattr(r.document, 'text'):
            results.append(r.document.text)
        elif r.document and isinstance(r.document, dict):
            results.append(r.document.get("text", ""))
        else:
            # fallback pakai index ke list original
            results.append(documents[r.index])

    return results