"""
MedAI – Database Models
PredictionLog stores every prediction for audit trail and history.
UserProfile extends Django User with health-related fields.
"""

from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    """Extended user profile with health and personal details."""

    GENDER_CHOICES = [
        ("male", "Male"),
        ("female", "Female"),
        ("other", "Other"),
        ("prefer_not_to_say", "Prefer not to say"),
    ]

    BLOOD_GROUP_CHOICES = [
        ("A+", "A+"), ("A-", "A-"),
        ("B+", "B+"), ("B-", "B-"),
        ("AB+", "AB+"), ("AB-", "AB-"),
        ("O+", "O+"), ("O-", "O-"),
        ("unknown", "Unknown"),
    ]

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    phone = models.CharField(max_length=15, blank=True, default="")
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES, blank=True, default="")
    blood_group = models.CharField(max_length=10, choices=BLOOD_GROUP_CHOICES, blank=True, default="unknown")
    height_cm = models.FloatField(null=True, blank=True, help_text="Height in centimeters")
    weight_kg = models.FloatField(null=True, blank=True, help_text="Weight in kilograms")
    allergies = models.TextField(blank=True, default="", help_text="Known allergies")
    medical_conditions = models.TextField(blank=True, default="", help_text="Existing medical conditions")
    emergency_contact = models.CharField(max_length=15, blank=True, default="")
    address = models.TextField(blank=True, default="")
    avatar_initial = models.CharField(max_length=2, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile: {self.user.username}"

    def save(self, *args, **kwargs):
        if not self.avatar_initial and self.user:
            name = self.user.get_full_name() or self.user.username
            self.avatar_initial = name[0].upper()
        super().save(*args, **kwargs)


class PredictionLog(models.Model):
    """Stores each prediction request and its result."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prediction_logs",
        null=True,
        blank=True,
    )
    symptoms = models.TextField(help_text="User-provided symptom description")
    predicted_disease = models.CharField(max_length=255)
    confidence = models.FloatField(help_text="Prediction confidence percentage")
    risk_level = models.CharField(max_length=50)
    is_emergency = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Prediction Log"
        verbose_name_plural = "Prediction Logs"

    def __str__(self):
        owner = self.user.username if self.user else "anonymous"
        return f"{owner}: {self.predicted_disease} ({self.confidence}%) – {self.created_at:%Y-%m-%d %H:%M}"
