"""
MedAI – API URL Routing
"""

from django.urls import path

from .views import (
    health_check,
    login_view,
    logout_view,
    predict_rag_view,
    predict_view,
    prediction_history,
    profile_view,
    register_view,
)

urlpatterns = [
    # ML endpoints
    path("predict/", predict_view, name="predict"),
    path("predict-rag/", predict_rag_view, name="predict-rag"),
    path("health/", health_check, name="health"),
    path("history/", prediction_history, name="history"),
    # Auth endpoints
    path("auth/register/", register_view, name="register"),
    path("auth/login/", login_view, name="login"),
    path("auth/logout/", logout_view, name="logout"),
    path("auth/profile/", profile_view, name="profile"),
]