from django.urls import path
from .views import GetUser, LoginView, RegisterView, UploadPDFView, ChatBot

urlpatterns = [
    # path('chat/', OllamaChatView.as_view(), name='ollama_chat'),
    path('upload-pdf/', UploadPDFView.as_view(), name='upload-pdf'),
    path('chat/', ChatBot.as_view(), name='chat'),
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('auth/login/', LoginView.as_view(), name='login'),
    path('auth/me/', GetUser.as_view(), name='get-user'),
]
