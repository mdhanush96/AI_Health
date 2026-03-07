"""
MedAI – Root URL Configuration
"""

from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def home(request):
    return JsonResponse({
        "application": "MedAI",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/api/predict/",
            "predict_rag": "/api/predict-rag/",
            "health": "/api/health/",
            "history": "/api/history/",
            "admin": "/admin/",
        },
    })


urlpatterns = [
    path("", home, name="home"),
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
]
