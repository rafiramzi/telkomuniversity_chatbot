from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import ollama  

class OllamaChatView(APIView):
    def post(self, request):
        user_message = request.data.get("message", "")

        if not user_message:
            return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = ollama.chat(
                model="gpt-oss:20b-cloud",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": user_message},
                ]
            )

            reply = result["message"]["content"]
            return Response({"reply": reply})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
