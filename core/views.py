# core/views.py
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework import status
import json, ollama

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
