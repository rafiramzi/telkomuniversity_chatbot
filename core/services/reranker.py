import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def rerank(query, documents, top_n=4):
    """
    Versi legacy yang hanya mengembalikan list teks dokumen
    (dipakai oleh kode lama). Tetap dipertahankan.
    """
    if not documents:
        return []

    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        return_documents=True,
    )

    print(f"[RERANK] results count: {len(response.results)}")
    for r in response.results:
        print(f"  index={r.index} | score={r.relevance_score:.4f} | doc={str(r.document)[:60]}")

    results = []
    for r in response.results:
        if r.document and hasattr(r.document, "text"):
            results.append(r.document.text)
        elif r.document and isinstance(r.document, dict):
            results.append(r.document.get("text", ""))
        else:
            results.append(documents[r.index])

    return results


def rerank_with_scores(query, documents, top_n=8):
    """
    Versi yang mengembalikan list of (text, score, original_index).
    Dipakai model2 supaya bisa filter berdasarkan relevance score —
    krusial karena Cohere rerank kasih score yang nyata: kalau cuma
    1 dokumen benar-benar relevan, scoring akan turun drastis di
    rank ke-2 ke bawah.
    """
    if not documents:
        return []

    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        return_documents=True,
    )

    print(f"[RERANK] results count: {len(response.results)}")
    for r in response.results:
        print(f"  index={r.index} | score={r.relevance_score:.4f} | doc={str(r.document)[:60]}")

    out = []
    for r in response.results:
        if r.document and hasattr(r.document, "text"):
            text = r.document.text
        elif r.document and isinstance(r.document, dict):
            text = r.document.get("text", "")
        else:
            text = documents[r.index]

        out.append({
            "text": text,
            "score": float(r.relevance_score),
            "index": r.index,
        })

    return out