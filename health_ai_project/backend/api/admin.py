"""
MedAI – Admin Configuration
"""

from django.contrib import admin

from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "predicted_disease",
        "confidence",
        "risk_level",
        "is_emergency",
        "created_at",
    ]
    list_filter = ["risk_level", "is_emergency", "created_at"]
    search_fields = ["symptoms", "predicted_disease"]
    readonly_fields = [
        "symptoms",
        "predicted_disease",
        "confidence",
        "risk_level",
        "is_emergency",
        "created_at",
    ]
    ordering = ["-created_at"]
