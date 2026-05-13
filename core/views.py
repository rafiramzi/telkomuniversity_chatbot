# core/views.py
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import json, ollama
import csv
import time
import chromadb
import pdfplumber
import os
import cohere
import numpy as np
from chromadb.api.types import EmbeddingFunction

from .services.vector_store import search
from .services.reranker import rerank
from .services.generator import generate_answer_stream
from django.http import StreamingHttpResponse
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from django.conf import settings


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
    
cohere_ef = CohereEmbeddingFunction(
    api_key=os.getenv("COHERE_API_KEY")
)

from .services.vector_store import get_collection

cohere_ef = CohereEmbeddingFunction(api_key=os.getenv("COHERE_API_KEY"))
collection = get_collection(cohere_ef)

class UploadPDFView(View):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        pdf_file = request.FILES.get("file")
        category = request.data.get("category")

        if not pdf_file or not category:
            return JsonResponse(
                {"error": "file and category are required"},
                status=400
            )

        # Save file
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, pdf_file.name)

        with open(file_path, "wb+") as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        # Extract text
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

        if not text.strip():
            return JsonResponse({"error": "PDF kosong"}, status=400)

        # 🔑 CHUNKING
        CHUNK_SIZE = 800
        chunks = [
            text[i:i + CHUNK_SIZE]
            for i in range(0, len(text), CHUNK_SIZE)
        ]

        ids = [f"{pdf_file.name}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "category": category,
                "source": pdf_file.name
            }
            for _ in chunks
        ]

        # ✅ ADD ONCE
        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )

        return JsonResponse({
            "message": "PDF berhasil di-embed",
            "chunks": len(chunks),
            "category": category
        })

    




def load_csv_data(file_path="dataset.csv"):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append({
                "id": row["file"],
                "category": row["category"],
                "text": row["text"],
                "created_at":row['created_at']
            })
    return docs


data = load_csv_data("dataset.csv")

for d in data:
    try:
        collection.add(
            ids=[d["id"]],
            documents=[d["text"]],
            metadatas=[{"category": d["category"]}]
        )
    except Exception:
        # Already exists
        pass



categories = list({d["category"] for d in data})
print(f"✅ Indexed {len(data)} documents into ChromaDB")
print(f"📂 Found categories: {categories}")
print(f"Using API Key: {os.getenv('COHERE_API_KEY')[:4]}...")


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
            # MODEL 1 — WITH CATEGORY
            # =========================
            if model == "model1":

                category = body.get("category")

                results = search(query=query, n_results=6, category=category)
                docs = results.get("documents", [[]])[0]
                context = "\n\n".join(docs) if docs else ""

                strict = False

            # =========================
            # MODEL 2 — WITH RERANK
            # =========================
            elif model == "model2":

                results = search(query=query, n_results=12)
                docs = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                print("\n" + "="*60)
                print(f"[MODEL2] QUERY: '{query}'")
                print(f"[MODEL2] Total docs ditemukan: {len(docs)}")
                print("-"*60)
                for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas)):
                    print(f"  [{i+1}] dist={dist:.4f} | category={meta.get('category','?')} | text={doc[:80]}...")
                print("-"*60)

                filtered_docs = [
                    d for d, dist in zip(docs, distances)
                    if dist < 0.6
                ]
                print(f"[MODEL2] Setelah filter dist<1.0: {len(filtered_docs)} docs tersisa")

                final_docs = rerank(query, filtered_docs, top_n=4) if filtered_docs else []
                print(f"[MODEL2] Setelah rerank top_n=4: {len(final_docs)} docs")
                for i, doc in enumerate(final_docs):
                    print(f"  [{i+1}] {doc[:100]}...")

                context = "\n\n".join(final_docs)
                print(f"[MODEL2] Context length: {len(context)} chars")
                print(f"[MODEL2] Context kosong: {not context.strip()}")
                print("="*60 + "\n")

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
                # Same message for both cases — don't leak which field was wrong
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