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
from django.conf import settings


from chromadb.utils import embedding_functions

class OllamaChatView(APIView):
    def post(self, request):
        message = request.data.get("message")
        model = request.data.get("model")

        if not message or not model:
            return StreamingHttpResponse(
                json.dumps({"error": "Message and model are required"}) + "\n",
                content_type="application/json",
                status=status.HTTP_400_BAD_REQUEST,
            )

        def stream():
            try:
                for chunk in ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    stream=True
                ):
                    if "message" in chunk and "content" in chunk["message"]:
                        yield json.dumps({"content": chunk["message"]["content"]}) + "\n"
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"

        return StreamingHttpResponse(stream(), content_type="application/x-ndjson")

class UploadPDFView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        """
        Upload a single PDF file with a category.
        Example form-data:
        - file: <uploaded.pdf>
        - category: "User Manual"
        """
        pdf_file = request.FILES.get('file')
        category = request.data.get('category')

        if not pdf_file or not category:
            return Response(
                {"error": "Both 'file' and 'category' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Save uploaded file temporarily
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, pdf_file.name)

        with open(file_path, "wb+") as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)

        # Extract text from PDF
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            return Response(
                {"error": f"Failed to extract text: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Append to CSV
        csv_file = os.path.join(settings.BASE_DIR, "dataset.csv")
        file_exists = os.path.exists(csv_file)
        header_exists = False

        if file_exists:
            with open(csv_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "file,category,text,created_at" in first_line:
                    header_exists = True

        with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["file", "category", "text", "created_at"])
            if not header_exists:
                writer.writeheader()
            writer.writerow({
                "file": pdf_file.name,
                "category": category,
                "text": text,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })

        return Response({
            "message": f"✅ File '{pdf_file.name}' processed and added to dataset.csv",
            "category": category,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }, status=status.HTTP_200_OK)
    

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

def load_csv_data(file_path="dataset.csv"):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append({
                "id": row["file"],
                "category": row["category"],
                "text": row["text"]
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

class ChatBot(APIView):
    def post(self, request):
        query = request.data.get("query", "").strip()
        if not query:
            return Response({"error": "Query text is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Detect category
        try:
            cat_query = ollama.chat(model="gpt-oss:20b-cloud", messages=[
                {"role": "system", "content": f"Available categories: {categories}"},
                {"role": "user", "content": f"Which category fits best for: {query}? Only answer with category name."}
            ])
            category_guess = cat_query["message"]["content"].strip()
            if category_guess not in categories:
                category_guess = categories[0]
        except Exception as e:
            category_guess = categories[0]
            print(f"Category detection failed: {e}")

        # Retrieve relevant context
        results = collection.query(
            query_texts=[query],
            n_results=3,
            where={"category": category_guess}
        )
        context = "\n\n".join(results["documents"][0]) if results["documents"] else "No relevant context found."

        # --- STREAM RESPONSE ---
        def stream_generator():
            start_time = time.time()
            try:
                stream = ollama.chat(
                    model="gpt-oss:20b-cloud",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a helpful assistant answering within '{category_guess}' category. you are not allowed to answer anything out of topic and out of the dataset. for greetings or conversation, tell the user that you are only answer about telkom university campus topic"
                                       f"Use this context:\n{context}"
                        },
                        {"role": "user", "content": query},
                    ],
                    stream=True,
                    options={"temperature":1.0}
                )

                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        # Each chunk is sent as a new line
                        yield chunk["message"]["content"]
                        # flush each piece immediately
                        yield ""
                elapsed = time.time() - start_time
                yield f"\n\n\n(answer time: {elapsed:.2f} seconds)"
            except Exception as e:
                yield f"\n\n[Error] {str(e)}"

        return StreamingHttpResponse(stream_generator(), content_type="text/plain")