from django.urls import path
from .views import UploadPDFView, ChatBot

urlpatterns = [
    # path('chat/', OllamaChatView.as_view(), name='ollama_chat'),
    path('upload-pdf/', UploadPDFView.as_view(), name='upload-pdf'),
    path('chat/', ChatBot.as_view(), name='chat'),

]
