"""
MedAI – API Serializers
"""

from django.contrib.auth.models import User
from rest_framework import serializers

from .models import PredictionLog, UserProfile


# ---------------------------------------------------------------------------
# Auth Serializers
# ---------------------------------------------------------------------------

class RegisterSerializer(serializers.Serializer):
    """User registration."""
    username = serializers.CharField(max_length=150, min_length=3)
    email = serializers.EmailField()
    password = serializers.CharField(min_length=6, write_only=True)
    first_name = serializers.CharField(max_length=150, required=False, default="")
    last_name = serializers.CharField(max_length=150, required=False, default="")

    def validate_username(self, value):
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("This username is already taken.")
        return value.lower()

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("An account with this email already exists.")
        return value.lower()

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
            first_name=validated_data.get("first_name", ""),
            last_name=validated_data.get("last_name", ""),
        )
        UserProfile.objects.create(user=user)
        return user


class LoginSerializer(serializers.Serializer):
    """User login."""
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class UserSerializer(serializers.ModelSerializer):
    """User basic info."""
    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name", "date_joined"]
        read_only_fields = fields


class UserProfileSerializer(serializers.ModelSerializer):
    """User profile with health details."""
    user = UserSerializer(read_only=True)

    class Meta:
        model = UserProfile
        fields = [
            "id", "user", "phone", "date_of_birth", "gender",
            "blood_group", "height_cm", "weight_kg", "allergies",
            "medical_conditions", "emergency_contact", "address",
            "avatar_initial", "created_at", "updated_at",
        ]
        read_only_fields = ["id", "user", "avatar_initial", "created_at", "updated_at"]


class UserProfileUpdateSerializer(serializers.Serializer):
    """Update user + profile fields."""
    # User fields
    first_name = serializers.CharField(max_length=150, required=False)
    last_name = serializers.CharField(max_length=150, required=False)
    email = serializers.EmailField(required=False)
    # Profile fields
    phone = serializers.CharField(max_length=15, required=False, allow_blank=True)
    date_of_birth = serializers.DateField(required=False, allow_null=True)
    gender = serializers.ChoiceField(choices=UserProfile.GENDER_CHOICES, required=False, allow_blank=True)
    blood_group = serializers.ChoiceField(choices=UserProfile.BLOOD_GROUP_CHOICES, required=False)
    height_cm = serializers.FloatField(required=False, allow_null=True)
    weight_kg = serializers.FloatField(required=False, allow_null=True)
    allergies = serializers.CharField(required=False, allow_blank=True)
    medical_conditions = serializers.CharField(required=False, allow_blank=True)
    emergency_contact = serializers.CharField(max_length=15, required=False, allow_blank=True)
    address = serializers.CharField(required=False, allow_blank=True)


# ---------------------------------------------------------------------------
# Symptom / Prediction Serializers
# ---------------------------------------------------------------------------

class SymptomInputSerializer(serializers.Serializer):
    """Validates incoming symptom prediction requests (text + optional file)."""
    symptoms = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=5000,
        help_text="Natural-language description of symptoms",
    )
    report = serializers.FileField(
        required=False,
        help_text="Optional medical report file (PDF, TXT, PNG, JPG)",
    )

    def validate(self, data):
        has_symptoms = bool(data.get("symptoms", "").strip())
        has_report = data.get("report") is not None
        if not has_symptoms and not has_report:
            raise serializers.ValidationError(
                "Please provide symptoms text or upload a medical report."
            )
        return data


class PredictionLogSerializer(serializers.ModelSerializer):
    """Serializes prediction history records."""

    class Meta:
        model = PredictionLog
        fields = [
            "id",
            "symptoms",
            "predicted_disease",
            "confidence",
            "risk_level",
            "is_emergency",
            "created_at",
        ]
        read_only_fields = fields