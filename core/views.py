# core/views.py
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import json, ollama
import time
import uuid
import chromadb
import pdfplumber
import os
import cohere
import numpy as np
from chromadb.api.types import EmbeddingFunction

from .services.vector_store import search
from .services.reranker import rerank
from .services.generator import generate_answer_stream
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



CHUNK_SIZE = 800

def warmup_chromadb_from_supabase():
    """Re-populate ChromaDB from Supabase on server start."""
    try:
        # Cek apakah ChromaDB sudah ada datanya
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

# Panggil di bawah inisialisasi collection
collection = get_collection(cohere_ef)
warmup_chromadb_from_supabase()  # ← tambahkan ini

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

        # ---- Chunk text ----
        chunks = [
            text[i:i + CHUNK_SIZE]
            for i in range(0, len(text), CHUNK_SIZE)
        ]

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
                        "dataset_id": chunk_ids[i],  # links back to Supabase row
                    }
                    for i in range(len(chunks))
                ],
            )
        except Exception as e:
            # Supabase already has the data - log and return partial success
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
USER_CATEGORY = {}   # kategori tiap session


@method_decorator(csrf_exempt, name='dispatch')
class ChatBot(View):
    renderer_classes = []

    def post(self, request):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except:
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
            # MODEL 1 - WITH CATEGORY
            # =========================
            if model == "model1":
                category = body.get("category")

                results = search(query=query, n_results=6, category=category)
                docs = results.get("documents", [[]])[0]
                context = "\n\n".join(docs) if docs else ""

                strict = False

            # =========================
            # MODEL 2 - WITH RERANK
            # =========================
            elif model == "model2":
                results = search(query=query, n_results=12)
                docs = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                print("\n" + "=" * 60)
                print(f"[MODEL2] QUERY: '{query}'")
                print(f"[MODEL2] Total docs ditemukan: {len(docs)}")
                print("-" * 60)
                for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas)):
                    print(f"  [{i+1}] dist={dist:.4f} | category={meta.get('category','?')} | text={doc[:80]}...")
                print("-" * 60)

                filtered_docs = [
                    d for d, dist in zip(docs, distances)
                    if dist < 0.6
                ]
                print(f"[MODEL2] Setelah filter dist<0.6: {len(filtered_docs)} docs tersisa")

                final_docs = rerank(query, filtered_docs, top_n=4) if filtered_docs else []
                print(f"[MODEL2] Setelah rerank top_n=4: {len(final_docs)} docs")
                for i, doc in enumerate(final_docs):
                    print(f"  [{i+1}] {doc[:100]}...")

                context = "\n\n".join(final_docs)
                print(f"[MODEL2] Context length: {len(context)} chars")
                print(f"[MODEL2] Context kosong: {not context.strip()}")
                print("=" * 60 + "\n")

                strict = True

            # =========================
            # STREAM FUNCTION
            # =========================
            def stream():
                print("STREAM STARTED")
                try:
                    for chunk in generate_answer_stream(query, context, strict=strict):
                        encoded = chunk.replace("\n", "\\n")
                        yield f"data: {encoded}\n\n"
                except Exception as e:
                    print("ERROR:", e)
                    yield f"data: [ERROR] {str(e)}\n\n"

            response = StreamingHttpResponse(
                stream(),
                content_type="text/event-stream",
            )

            response["Cache-Control"] = "no-cache"
            response["X-Accel-Buffering"] = "no"

            return response

        except Exception as e:
            return JsonResponse(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# =============================================================================
# AUTH
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

        # Validation
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
            # Check if user already exists
            existing = supabase.table("users").select("id").or_(
                f"username.eq.{username},email.eq.{email}"
            ).execute()

            if existing.data:
                return JsonResponse(
                    {"error": "Username or email already registered"},
                    status=409
                )

            # Hash password with bcrypt
            password_bytes = password.encode("utf-8")
            hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=12))
            hashed_str = hashed.decode("utf-8")

            # Insert into Supabase
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
            # Look up user by username OR email
            result = supabase.table("users").select("*").or_(
                f"username.eq.{identifier},email.eq.{identifier}"
            ).limit(1).execute()

            if not result.data:
                return JsonResponse(
                    {"error": "Invalid credentials"},
                    status=401
                )

            user = result.data[0]

            # Verify password
            password_bytes = password.encode("utf-8")
            stored_hash = user["password"].encode("utf-8")

            if not bcrypt.checkpw(password_bytes, stored_hash):
                return JsonResponse(
                    {"error": "Invalid credentials"},
                    status=401
                )

            # Generate JWT token
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
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JsonResponse({"error": "Missing or invalid Authorization header"}, status=401)

        token = auth_header[7:]  # strip "Bearer "

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