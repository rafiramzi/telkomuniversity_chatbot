# core/views.py
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import json, ollama
import time
import re
import uuid
import chromadb
import pdfplumber
import os
import cohere
import numpy as np
from chromadb.api.types import EmbeddingFunction

from .services.vector_store import search
from .services.reranker import rerank, rerank_with_scores
from .services.generator import (
    generate_answer_stream,
    generate_grounded_answer_stream,
)
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from django.conf import settings


from chromadb.utils import embedding_functions
from supabase import create_client

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

class CohereEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model="embed-multilingual-v3.0"):
        self.co = cohere.Client(api_key)
        self.model = model

    def __call__(self, texts):
        response = self.co.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings


from .services.vector_store import get_collection

cohere_ef = CohereEmbeddingFunction(api_key=os.getenv("COHERE_API_KEY"))
collection = get_collection(cohere_ef)

print(f"Using API Key: {os.getenv('COHERE_API_KEY')[:4]}...")



# Chunk lebih kecil + overlap supaya satu konsep tidak terpotong di tengah.
# Claude version pakai 400/50 dan hasilnya lebih akurat.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Threshold cosine distance (0..1). Lebih kecil = lebih ketat.
# 0.55 cukup permisif untuk Bahasa Indonesia + dokumen akademik.
MODEL2_DISTANCE_THRESHOLD = 0.55

# Stopwords Indonesia ringan untuk keyword boosting
_ID_STOPWORDS = {
    "yang", "dan", "atau", "untuk", "dengan", "dalam", "adalah", "pada",
    "apa", "bagaimana", "berapa", "tentang", "itu", "ini", "saya", "saja",
    "akan", "dari", "ke", "di", "ada", "tidak", "juga", "bisa", "dapat",
    "agar", "supaya", "oleh", "telah", "sudah",
}


def _chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sliding-window chunker dengan overlap kecil."""
    chunks = []
    step = size - overlap
    if step <= 0:
        step = size
    for i in range(0, len(text), step):
        piece = text[i:i + size]
        if piece.strip():
            chunks.append(piece)
    return chunks


def _keyword_boost(query, doc_text):
    """
    Boost skor dokumen yang mengandung acronym/keyword penting dari query.
    Krusial buat akademik Indonesia karena IPK vs IPS vs IKK vs TAK
    berbeda arti tapi mirip embedding-nya.

    Strategi:
    - Acronym exact-match diberi boost besar (+0.30 per acronym).
    - Kata kunci panjang dapat boost kecil (+0.05).
    - Acronym query yang TIDAK muncul di dokumen → kandidat tetap, tapi
      tidak mendapat boost. Ranking semantic tetap berlaku sebagai dasar.
    """
    boost = 0.0

    text_upper = doc_text.upper()
    text_lower = doc_text.lower()

    # Acronym (≥2 huruf kapital, mis. IPK, SKS, TAK).
    # Boost besar karena ini sinyal paling andal untuk dokumen akademik.
    acronyms = set(re.findall(r"\b[A-Z]{2,}\b", query.upper()))
    for ac in acronyms:
        if re.search(r"\b" + re.escape(ac) + r"\b", text_upper):
            boost += 0.30

    # Kata kunci panjang (≥4 char, bukan stopword)
    keywords = {
        w.lower() for w in re.findall(r"\b\w+\b", query)
        if len(w) >= 4 and w.lower() not in _ID_STOPWORDS
    }
    for kw in keywords:
        if kw in text_lower:
            boost += 0.05

    return boost


# Pattern untuk mendeteksi query yang minta enumerasi / daftar lengkap.
# Untuk query semacam ini, diversitas chunk lebih penting daripada
# precision tinggi — kalau filtering terlalu ketat, jawabannya jadi
# sebagian (mis. cuma 1 dari 5 lokasi yang disebutkan).
_ENUMERATION_PATTERNS = [
    r"\bapa\s+saja\b",
    r"\bsemua\b",
    r"\bdaftar\b",
    r"\bsebutkan\b",
    r"\bjelaskan\s+semua\b",
    r"\bberapa\s+banyak\b",
    r"\bada\s+berapa\b",
    r"\bmana\s+saja\b",
    r"\bsiapa\s+saja\b",
    # Reduplikasi Bahasa Indonesia: "lokasi-lokasi", "kampus-kampus"
    r"\b(\w{3,})-\1\b",
    # Plural/list cues
    r"\b(list|daftar|kumpulan)\b",
]


def _is_enumeration_query(query: str) -> bool:
    """Deteksi query yang minta enumerasi/daftar lengkap."""
    q = query.lower()
    for pat in _ENUMERATION_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def warmup_chromadb_from_supabase():
    """Re-populate ChromaDB from Supabase on server start."""
    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"[WARMUP] ChromaDB sudah ada {existing_count} docs, skip.")
            return

        print("[WARMUP] ChromaDB kosong, loading dari Supabase...")
        result = supabase.table("datasets").select("*").execute()
        rows = result.data or []

        if not rows:
            print("[WARMUP] Supabase juga kosong.")
            return

        ids = [str(r["id"]) for r in rows]
        texts = [r["text"] for r in rows]
        metadatas = [{"category": r["category"], "source": r["file"]} for r in rows]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        print(f"[WARMUP] Berhasil load {len(rows)} docs dari Supabase ke ChromaDB.")

    except Exception as e:
        print(f"[WARMUP ERROR] {e}")


collection = get_collection(cohere_ef)
warmup_chromadb_from_supabase()


@method_decorator(csrf_exempt, name='dispatch')
class UploadPDFView(View):
    def post(self, request):
        pdf_file = request.FILES.get("file")
        category = request.POST.get("category")

        if not pdf_file or not category:
            return JsonResponse(
                {"error": "file and category are required"},
                status=400
            )

        # ---- Save file to disk ----
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, pdf_file.name)

        with open(file_path, "wb+") as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        # ---- Extract text ----
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            return JsonResponse({"error": "PDF kosong"}, status=400)

        # ---- Chunk text (sliding window dengan overlap) ----
        chunks = _chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        # Shared UUIDs - same id in both Supabase and ChromaDB
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        # ---- 1) Insert into Supabase (source of truth) ----
        try:
            rows = [
                {
                    "id": chunk_ids[i],
                    "file": pdf_file.name,
                    "category": category,
                    "text": chunk_text,
                }
                for i, chunk_text in enumerate(chunks)
            ]

            result = supabase.table("datasets").insert(rows).execute()

            if not result.data:
                return JsonResponse(
                    {"error": "Failed to insert into Supabase"},
                    status=500
                )
        except Exception as e:
            return JsonResponse(
                {"error": f"Supabase insert failed: {str(e)}"},
                status=500
            )

        # ---- 2) Add to ChromaDB (vector index) ----
        try:
            collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=[
                    {
                        "category": category,
                        "source": pdf_file.name,
                        "dataset_id": chunk_ids[i],
                    }
                    for i in range(len(chunks))
                ],
            )
        except Exception as e:
            print(f"WARNING: ChromaDB indexing failed: {e}")
            return JsonResponse({
                "message": "Saved to database, but vector indexing failed",
                "chunks": len(chunks),
                "category": category,
                "warning": str(e),
            }, status=207)

        return JsonResponse({
            "message": "PDF berhasil di-embed",
            "chunks": len(chunks),
            "category": category,
            "file": pdf_file.name,
        })


# =============================================================================
# CHATBOT
# =============================================================================
CONVERSATION_MEMORY = {}
MAX_MEMORY = 2
USER_CATEGORY = {}


@method_decorator(csrf_exempt, name='dispatch')
class ChatBot(View):
    renderer_classes = []

    def post(self, request):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        model = body.get("model")
        query = body.get("query", "").strip()

        if not query:
            return JsonResponse(
                {"error": "Query text is required."},
                status=400
            )

        try:
            # =========================
            # MODEL 1 - WITH CATEGORY (tidak diubah)
            # =========================
            if model == "model1":
                category = body.get("category")

                results = search(query=query, n_results=6, category=category)
                docs = results.get("documents", [[]])[0]
                context = "\n\n".join(docs) if docs else ""
                strict = False

                def stream():
                    print("STREAM STARTED (model1)")
                    try:
                        for chunk in generate_answer_stream(query, context, strict=strict):
                            encoded = chunk.replace("\n", "\\n")
                            yield f"data: {encoded}\n\n"
                    except Exception as e:
                        print("ERROR:", e)
                        yield f"data: [ERROR] {str(e)}\n\n"

                response = StreamingHttpResponse(stream(), content_type="text/event-stream")
                response["Cache-Control"] = "no-cache"
                response["X-Accel-Buffering"] = "no"
                return response

            # =========================
            # MODEL 2 - GROUNDED RAG dengan rerank & keyword boost
            # =========================
            elif model == "model2":
                # ----------------------------------------------------------
                # Cek chitchat SEBELUM vector search.
                # Kalau vector store kosong (0 kandidat), stream_empty()
                # dipanggil sebelum generate_grounded_answer_stream,
                # sehingga _is_chitchat di generator tidak pernah tercapai.
                # Solusi: intercept di sini, langsung pakai generate_answer_stream
                # tanpa dokumen.
                # ----------------------------------------------------------
                from .services.generator import _is_chitchat
                if _is_chitchat(query):
                    def stream_chitchat():
                        from .services.generator import _CHITCHAT_SYSTEM, co_v2, _sanitize_stream
                        try:
                            s = co_v2.chat_stream(
                                model="command-a-03-2025",
                                messages=[
                                    {"role": "system", "content": _CHITCHAT_SYSTEM},
                                    {"role": "user", "content": query},
                                ],
                                temperature=0.7,
                            )
                            def _raw():
                                for event in s:
                                    if event.type == "content-delta":
                                        try:
                                            text = event.delta.message.content.text
                                            if text:
                                                yield text
                                        except Exception:
                                            pass
                            for chunk in _sanitize_stream(_raw()):
                                encoded = chunk.replace("\n", "\\n")
                                yield f"data: {encoded}\n\n"
                        except Exception as e:
                            yield f"data: [ERROR] {str(e)}\n\n"

                    resp = StreamingHttpResponse(stream_chitchat(), content_type="text/event-stream")
                    resp["Cache-Control"] = "no-cache"
                    resp["X-Accel-Buffering"] = "no"
                    return resp

                # Deteksi jenis query: enumerasi vs spesifik.
                is_enum = _is_enumeration_query(query)

                # Tuning per jenis query
                if is_enum:
                    retrieve_k = 20         # Ambil lebih banyak untuk diversitas
                    rerank_top_n = 12
                    min_rerank_score = 0.05  # Lebih permisif
                    max_final = 8            # Pass lebih banyak ke LLM
                else:
                    retrieve_k = 15
                    rerank_top_n = 8
                    min_rerank_score = 0.10
                    max_final = 5

                # 1) Ambil kandidat luas supaya rerank punya bahan
                results = search(query=query, n_results=retrieve_k)
                docs = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                ids = results.get("ids", [[]])[0]

                print("\n" + "=" * 60)
                print(f"[MODEL2] QUERY: '{query}'")
                print(f"[MODEL2] Enumeration query? {is_enum} "
                      f"(retrieve_k={retrieve_k}, max_final={max_final}, "
                      f"min_score={min_rerank_score})")
                print(f"[MODEL2] Total kandidat: {len(docs)}")
                print("-" * 60)
                for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas)):
                    print(f"  [{i+1}] dist={dist:.4f} | cat={meta.get('category','?')} | {doc[:70].strip()}...")

                # 2) Filter berdasarkan cosine distance (sekarang sudah benar
                #    karena collection di-set hnsw:space=cosine, jadi 0..1).
                #    Kalau semua kandidat lolos threshold ketat, tetap ambil
                #    minimal 6 supaya rerank/grounding tidak kelaparan.
                filtered = [
                    {"id": ids[i], "text": docs[i], "dist": distances[i], "meta": metadatas[i]}
                    for i in range(len(docs))
                    if distances[i] < MODEL2_DISTANCE_THRESHOLD
                ]

                if len(filtered) < 6:
                    # fallback: ambil 6 terbaik apa pun jaraknya
                    filtered = [
                        {"id": ids[i], "text": docs[i], "dist": distances[i], "meta": metadatas[i]}
                        for i in range(min(len(docs), 6))
                    ]
                    print(f"[MODEL2] Threshold terlalu ketat, fallback ke top-{len(filtered)}")
                else:
                    print(f"[MODEL2] Setelah filter dist<{MODEL2_DISTANCE_THRESHOLD}: {len(filtered)} docs")

                if not filtered:
                    # Benar-benar kosong → vector store belum ada data
                    def stream_empty():
                        yield "data: Maaf, informasi tersebut tidak tersedia dalam data yang saya miliki.\n\n"
                    resp = StreamingHttpResponse(stream_empty(), content_type="text/event-stream")
                    resp["Cache-Control"] = "no-cache"
                    resp["X-Accel-Buffering"] = "no"
                    return resp

                # 3) Hybrid score: semantic similarity + keyword/acronym boost
                #    (cosine distance → similarity = 1 - dist)
                for item in filtered:
                    sim = 1.0 - item["dist"]
                    item["score"] = sim + _keyword_boost(query, item["text"])
                filtered.sort(key=lambda x: x["score"], reverse=True)

                print("-" * 60)
                print(f"[MODEL2] Top 5 setelah keyword boost:")
                for i, item in enumerate(filtered[:5]):
                    print(f"  [{i+1}] score={item['score']:.4f} | {item['text'][:70].strip()}...")

                # 4) Cohere rerank di top-N kandidat. Yang penting di sini
                #    BUKAN sekedar "top N" — tapi memilih chunk yang
                #    relevance_score-nya cukup tinggi. Kalau rerank bilang
                #    cuma 1 chunk yang relevan (score >> sisanya), passing
                #    5 chunk ke LLM justru meracuni konteks dengan noise.
                top_candidates = filtered[: max(10, retrieve_k - 5)]
                texts_for_rerank = [c["text"] for c in top_candidates]

                try:
                    reranked = rerank_with_scores(query, texts_for_rerank, top_n=rerank_top_n)
                except Exception as rerank_err:
                    print(f"[MODEL2] Rerank gagal, pakai urutan keyword-boost: {rerank_err}")
                    reranked = [
                        {"text": c["text"], "score": 0.5, "index": i}
                        for i, c in enumerate(top_candidates[:5])
                    ]

                # Threshold absolut. Score Cohere rerank-multilingual-v3 biasanya:
                #   > 0.5  = sangat relevan, direct match
                #   0.1-0.5 = relevan, kemungkinan berisi informasi terkait
                #   < 0.1  = kurang relevan, kemungkinan noise
                # Untuk enumeration query, threshold diturunkan supaya
                # chunk dari topik berbeda tetap masuk konteks.
                strong_matches = [r for r in reranked if r["score"] >= min_rerank_score]

                # Kalau threshold menyaring semuanya, ambil saja top-1
                # supaya LLM masih punya bahan untuk menyebut topik terkait.
                if not strong_matches and reranked:
                    strong_matches = reranked[:1]
                    print(f"[MODEL2] Semua skor di bawah {min_rerank_score}, "
                          f"fallback ke top-1 (score={reranked[0]['score']:.4f})")
                else:
                    print(f"[MODEL2] {len(strong_matches)} chunks lolos "
                          f"threshold rerank ≥ {min_rerank_score}")

                # Map balik ke item lengkap (id + meta) via index original
                final_chunks = []
                seen_ids = set()
                categories_seen = set()
                for r in strong_matches:
                    item = top_candidates[r["index"]]
                    if item["id"] in seen_ids:
                        continue
                    seen_ids.add(item["id"])
                    final_chunks.append({
                        "id": item["id"],
                        "text": item["text"],
                        "category": item["meta"].get("category", ""),
                        "source": item["meta"].get("source", ""),
                        "rerank_score": r["score"],
                    })
                    categories_seen.add(item["meta"].get("category", ""))

                # Untuk enumeration query, kalau hasil rerank dominan dari
                # SATU kategori, tambahkan chunk dari kategori lain (dari
                # filtered list) supaya jawaban tidak miskin diversitas.
                # Contoh: query "lokasi kampus" dapat 5 chunk Jakarta saja —
                # tambahin chunk Bandung/Surabaya/Purwokerto kalau ada.
                if is_enum and len(categories_seen) <= 2 and len(final_chunks) < max_final:
                    print(f"[MODEL2] Enum query, kategori baru = {categories_seen}, "
                          f"tambahkan chunk dari kategori lain")
                    for item in filtered:
                        if len(final_chunks) >= max_final:
                            break
                        if item["id"] in seen_ids:
                            continue
                        cat = item["meta"].get("category", "")
                        if cat in categories_seen:
                            continue
                        # Hanya tambahkan kalau score semantic-nya masih oke
                        if item.get("score", 0) < 0.4:
                            continue
                        seen_ids.add(item["id"])
                        categories_seen.add(cat)
                        final_chunks.append({
                            "id": item["id"],
                            "text": item["text"],
                            "category": cat,
                            "source": item["meta"].get("source", ""),
                            "rerank_score": 0.0,  # ditambahkan untuk diversitas
                        })
                        print(f"  + diversity chunk: cat={cat} | {item['text'][:60].strip()}...")

                # Trim ke batas akhir
                final_chunks = final_chunks[:max_final]

                # Safety net (seharusnya tidak pernah trigger karena
                # ada fallback top-1 di atas)
                if not final_chunks:
                    final_chunks = [
                        {
                            "id": c["id"],
                            "text": c["text"],
                            "category": c["meta"].get("category", ""),
                            "source": c["meta"].get("source", ""),
                            "rerank_score": 0.0,
                        }
                        for c in filtered[:3]
                    ]

                # Indikator confidence untuk preamble: kalau best score < 0.3,
                # kasih tahu LLM bahwa match-nya lemah supaya jawabannya
                # eksplisit "informasi spesifik tidak ditemukan" + sebutkan
                # topik terkait yang tersedia.
                best_score = max((c["rerank_score"] for c in final_chunks), default=0.0)
                weak_match = best_score < 0.3

                print(f"[MODEL2] Final chunks dikirim ke LLM: {len(final_chunks)} "
                      f"(best_rerank_score={best_score:.4f}, weak_match={weak_match}, "
                      f"categories={len(categories_seen)})")
                print("=" * 60 + "\n")

                # 5) Grounded RAG streaming (Cohere command-r-plus dengan documents=)
                def stream():
                    print("STREAM STARTED (model2 grounded)")
                    try:
                        for chunk in generate_grounded_answer_stream(
                            query, final_chunks,
                            weak_match=weak_match,
                            is_enumeration=is_enum,
                        ):
                            encoded = chunk.replace("\n", "\\n")
                            yield f"data: {encoded}\n\n"
                    except Exception as e:
                        print("ERROR:", e)
                        yield f"data: [ERROR] {str(e)}\n\n"

                response = StreamingHttpResponse(stream(), content_type="text/event-stream")
                response["Cache-Control"] = "no-cache"
                response["X-Accel-Buffering"] = "no"
                return response

            else:
                return JsonResponse(
                    {"error": f"Unknown model '{model}'. Use 'model1' or 'model2'."},
                    status=400,
                )

        except Exception as e:
            return JsonResponse(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# =============================================================================
# AUTH (tidak diubah)
# =============================================================================
@method_decorator(csrf_exempt, name='dispatch')
class RegisterView(View):
    def post(self, request):
        try:
            data = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not username or not email or not password:
            return JsonResponse(
                {"error": "username, email, and password are required"},
                status=400
            )

        if len(password) < 8:
            return JsonResponse(
                {"error": "Password must be at least 8 characters"},
                status=400
            )

        try:
            existing = supabase.table("users").select("id").or_(
                f"username.eq.{username},email.eq.{email}"
            ).execute()

            if existing.data:
                return JsonResponse(
                    {"error": "Username or email already registered"},
                    status=409
                )

            password_bytes = password.encode("utf-8")
            hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=12))
            hashed_str = hashed.decode("utf-8")

            result = supabase.table("users").insert({
                "username": username,
                "email": email,
                "password": hashed_str,
            }).execute()

            if not result.data:
                return JsonResponse(
                    {"error": "Failed to create user"},
                    status=500
                )

            user = result.data[0]
            return JsonResponse({
                "message": "User registered successfully",
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                }
            }, status=201)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class LoginView(View):
    def post(self, request):
        try:
            data = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        identifier = data.get("username") or data.get("email", "")
        identifier = identifier.strip().lower() if identifier else ""
        password = data.get("password", "")

        if not identifier or not password:
            return JsonResponse(
                {"error": "username/email and password are required"},
                status=400
            )

        try:
            result = supabase.table("users").select("*").or_(
                f"username.eq.{identifier},email.eq.{identifier}"
            ).limit(1).execute()

            if not result.data:
                return JsonResponse(
                    {"error": "Invalid credentials"},
                    status=401
                )

            user = result.data[0]

            password_bytes = password.encode("utf-8")
            stored_hash = user["password"].encode("utf-8")

            if not bcrypt.checkpw(password_bytes, stored_hash):
                return JsonResponse(
                    {"error": "Invalid credentials"},
                    status=401
                )

            payload = {
                "user_id": user["id"],
                "username": user["username"],
                "exp": datetime.now(timezone.utc) + timedelta(days=7),
                "iat": datetime.now(timezone.utc),
            }
            token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

            return JsonResponse({
                "message": "Login successful",
                "token": token,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                }
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class GetUser(View):
    def get(self, request):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JsonResponse({"error": "Missing or invalid Authorization header"}, status=401)

        token = auth_header[7:]

        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            user_id = payload.get("user_id")
        except jwt.ExpiredSignatureError:
            return JsonResponse({"error": "Token has expired"}, status=401)
        except jwt.InvalidTokenError:
            return JsonResponse({"error": "Invalid token"}, status=401)

        try:
            result = supabase.table("users").select("id, username, email, user_role").eq("id", user_id).limit(1).execute()
            if not result.data:
                return JsonResponse({"error": "User not found"}, status=404)

            user = result.data[0]
            return JsonResponse({
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["user_role"],
                }
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)