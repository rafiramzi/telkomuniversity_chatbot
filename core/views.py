# core/views.py
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
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


from django.conf import settings


from chromadb.utils import embedding_functions

# class OllamaChatView(APIView):
#     def post(self, request):
#         message = request.data.get("message")
#         model = request.data.get("model")

#         if not message or not model:
#             return StreamingHttpResponse(
#                 json.dumps({"error": "Message and model are required"}) + "\n",
#                 content_type="application/json",
#                 status=status.HTTP_400_BAD_REQUEST,
#             )

#         def stream():
#             try:
#                 for chunk in ollama.chat(
#                     model=model,
#                     messages=[{"role": "user", "content": message}],
#                     stream=True
#                 ):
#                     if "message" in chunk and "content" in chunk["message"]:
#                         yield json.dumps({"content": chunk["message"]["content"]}) + "\n"
#             except Exception as e:
#                 yield json.dumps({"error": str(e)}) + "\n"

#         return StreamingHttpResponse(stream(), content_type="application/x-ndjson")

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
class UploadPDFView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        pdf_file = request.FILES.get("file")
        category = request.data.get("category")

        if not pdf_file or not category:
            return Response(
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
            return Response({"error": "PDF kosong"}, status=400)

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

        return Response({
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

class ChatBot(APIView):

    ai_model = "gemini-3-flash-preview:cloud"
    co = cohere.Client("NUmC8c9SPzKzoI8CW00zAA8L6SjCcHkIvTMqip2x")

    def post(self, request):
        model = request.data.get("model")




        # --- MODEL 1: WITH CATEGORY, NO RERANK ---
        if model == "model1":
                session_id = request.data.get("session_id", "default_user")
                # --- GET QUERY ---
                query = request.data.get("query", "").strip()
                if not query:
                    return Response({"error": "Query text is required."},
                                    status=status.HTTP_400_BAD_REQUEST)

                # --- DETECT CATEGORY ---
                try:
                    cat_query = ollama.chat(
                        model=self.ai_model,
                        messages=[
                            {"role": "system", "content": f"Available categories: {categories}"},
                            {"role": "user", "content": f"Which category fits best for: {query}? Only answer with category name. if the answer is not related about Telkom University Campus, the category is not relevant. Normal conversation is allowed"}
                        ],
                        options = { "temperature":1.0 }
                    )
                    category_guess = cat_query["message"]["content"].strip()
                except:
                    category_guess = "Not Relevant"

                # --- CHECK IF TOPIC CHANGED ---
                prev_cat = USER_CATEGORY.get(session_id)

                if prev_cat != category_guess:
                    # Reset history if topic changes
                    CONVERSATION_MEMORY[session_id] = []
                    USER_CATEGORY[session_id] = category_guess

                # --- UPDATE HISTORY AFTER RESET ---
                history = CONVERSATION_MEMORY.get(session_id, [])
                history.append({"role": "user", "content": query})
                if len(history) > MAX_MEMORY:
                    history = history[-MAX_MEMORY:]

                CONVERSATION_MEMORY[session_id] = history

                # --- VECTOR SEARCH ---
                results = collection.query(
                    query_texts=[query],
                    n_results=3,
                    where={"category": category_guess}
                )
                context = "\n\n".join(results["documents"][0]) if results["documents"] else "No relevant context found."

                # --- STREAM RESPONSE ---
                def stream_generator():
                    try:
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    f"You are a helpful assistant. Current category: '{category_guess}'. you are not allowed to answer anything out of topic and out of the dataset. for greetings or conversation, tell the user that you are only answer about telkom university campus topic. if category is Not Relevant, recomend the user to ask anything else about national campus\n"
                                    f"If user goes off-topic, inform them.\n\n"
                                    f"Context:\n{context}"
                                )
                            }
                        ] + history

                        for chunk in ollama.chat(
                            model=self.ai_model,
                            messages=messages,
                            stream=True,
                            options={"temperature": 0.9} 
                        ):
                            if "message" in chunk and "content" in chunk["message"]:
                                yield chunk["message"]["content"]
                                yield ""
                    except Exception as e:
                        yield f"\n\n[Error] {str(e)}"

                return StreamingHttpResponse(stream_generator(), content_type="text/plain")





        # --- MODEL 2: NO CATEGORY, WITH RERANK ---
        if model == "model2":
            query = request.data.get("query", "").strip()
            if not query:
                return Response({"error": "Query text is required."},
                                status=status.HTTP_400_BAD_REQUEST)

            # --- VECTOR SEARCH (no category filtering) ---
            results = collection.query(
                query_texts=[query],
                n_results=12,
                include=["documents", "distances"]
            )

            docs = results["documents"][0]
            distances = results["distances"][0]

            filtered_docs = [
                d for d, dist in zip(docs, distances)
                if dist < 0.10  # TUNING
            ]


            # --- OPTIONAL: Cohere Rerank ---
            final_docs = []

            if filtered_docs:
                # Rerank membutuhkan format {"text": "..."}
                docs_for_rerank = [{"text": d} for d in filtered_docs]

                reranked = self.co.rerank(
                    model="rerank-multilingual-v3.0",
                    query=query,
                    documents=docs_for_rerank,
                    top_n=4
                )

                # Ambil hasil yang valid
                final_docs = [
                    item.document["text"]
                    for item in reranked.results
                    if item.document and "text" in item.document
                ]

            # Fallback jika rerank kosong
            if not final_docs and filtered_docs:
                final_docs = filtered_docs[:4]


            # Fallback total
            context = "\n\n".join(final_docs) if final_docs else "No relevant context found."

            # --- STREAM RESPONSE ---
            def stream_generator():
                try:
                    system_msg = f"""
                    You are a question-answering system for Telkom University.

                    ALLOWED :
                    - Normal conversation is allowed but always steer back to campus-related topics.
                    
                    STRICT RULES:
                    - You MUST answer using ONLY the information explicitly stated in the context.
                    - DO NOT use prior knowledge.
                    - DO NOT make assumptions.
                    - DO NOT combine context with external knowledge.
                    - If the answer is NOT clearly stated in the context, respond EXACTLY with:

                    "Maaf, informasi tersebut tidak tersedia dalam data yang saya miliki."

                    Context:
                    {context}
                    """

                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": query}
                    ]

                    for chunk in ollama.chat(
                        model=self.ai_model,
                        messages=messages,
                        stream=True,
                        options={"temperature": 0.2}
                    ):
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                            yield ""
                except Exception as e:
                    yield f"\n\n[Error] {str(e)}"

            return StreamingHttpResponse(stream_generator(), content_type="text/plain")
        return Response(
            {"error": "Invalid model specified."},
            status=status.HTTP_400_BAD_REQUEST
        )
