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


from django.conf import settings


from chromadb.utils import embedding_functions

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
    api_key="NUmC8c9SPzKzoI8CW00zAA8L6SjCcHkIvTMqip2x"
)

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="pdf_docs_cohere",
    embedding_function=cohere_ef
)
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

                filtered_docs = [
                    d for d, dist in zip(docs, distances)
                    if dist < 0.15
                ]

                final_docs = rerank(query, filtered_docs, top_n=4) if filtered_docs else []
                context = "\n\n".join(final_docs)
                strict = True

            else:
                return JsonResponse(
                    {"error": "Invalid model"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # =========================
            # STREAM FUNCTION
            # =========================
            def stream():
                print("STREAM STARTED")
                try:
                    for chunk in generate_answer_stream(query, context, strict=strict):
                        print("CHUNK:", chunk)
                        yield f"data: {chunk}\n\n"
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