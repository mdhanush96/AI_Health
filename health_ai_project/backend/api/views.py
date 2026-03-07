"""
MedAI – API Views
Production endpoints for prediction, RAG query, health check, and history.
Supports both text symptoms and uploaded medical report files.
"""

import logging
import re

from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from ml_engine.predictor import predict, predict_with_rag
from ml_engine.report_summarizer import is_summary_request, summarize_report
from ml_engine.utils import extract_text_from_file, get_gpu_info, sanitize_input

from .models import PredictionLog, UserProfile
from .serializers import (
    LoginSerializer,
    PredictionLogSerializer,
    RegisterSerializer,
    SymptomInputSerializer,
    UserProfileSerializer,
    UserProfileUpdateSerializer,
)

logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Greeting / Non-Medical Input Detection
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = [
    # Pure greetings
    r"^\s*(hi|hello|hey|hii+|hola|namaste|namaskar)\s*[!.,]?\s*$",
    r"^\s*(good\s*(morning|afternoon|evening|night|day))\s*[!.,]?\s*$",

    # Greeting + intro
    r"^\s*(hi|hello|hey|hii+)\b.*\b(i\s*am|i'm|my\s*name\s*is|this\s*is)\b",

    # Greeting + capability question (e.g. "hello what can you do for me?")
    r"^\s*(hi|hello|hey)\s*[,!.]?\s*(what\s+can\s+you|how\s+can\s+you|can\s+you\s+help|help\s+me)",

    # Small talk
    r"^\s*(how\s*are\s*you|what'?\s*s?\s*up|wassup|sup)\s*[!?.,]?\s*$",

    # Thanks / bye
    r"^\s*(thanks?|thank\s*you|ty|thx)\s*[!.,]?\s*$",
    r"^\s*(bye|goodbye|see\s*you|take\s*care)\s*[!.,]?\s*$",

    # Identity questions
    r"^\s*(who\s*are\s*you|what\s*are\s*you|what\s*is\s*your\s*name)\s*[!?.,]?\s*$",
    r"^\s*(i\s*am|i'm|my\s*name\s*is)\s+\w+\s*[!.,]?\s*$",

    # Capability / help questions (standalone — no greeting prefix needed)
    r"^\s*what\s+can\s+you\s+do",
    r"^\s*what\s+do\s+you\s+do",
    r"^\s*how\s+can\s+you\s+help",
    r"^\s*how\s+do\s+you\s+work",
    r"^\s*what\s+are\s+you(r)?\s+(capabilit|feature|function)",
    r"^\s*help\s*me\s*[!?.,]?\s*$",
    r"^\s*what\s+(?:services?|things?)\s+(?:do\s+you|can\s+you)\s+(?:offer|provide|do)",
    r"^\s*can\s+you\s+help\s+me",
    r"^\s*i\s+need\s+help\s*[!?.,]?\s*$",
    r"^\s*tell\s+me\s+about\s+yourself",
]

_GREETING_RESPONSES = {
    "greeting": (
        "Hello! 👋 I'm MedAI, your health assistant. "
        "I'm here to help you understand your symptoms and guide you toward the right care.\n\n"
        "You can describe how you're feeling — for example:\n"
        "• \"I have a headache and mild fever\"\n"
        "• \"I've been feeling chest discomfort\"\n\n"
        "How can I help you today?"
    ),
    "capability": (
        "Great question! 😊 Here's what I can do:\n\n"
        "🩺 **Symptom Analysis** — Describe your symptoms and I'll identify possible conditions.\n"
        "📄 **Report Summarization** — Upload a medical report (PDF) and I'll summarize it clearly.\n"
        "💊 **Health Guidance** — I provide info on causes, treatments, and when to see a doctor.\n\n"
        "Try it out! Just describe how you're feeling, or upload a medical report."
    ),
    "thanks": (
        "You're welcome! 😊 I'm glad I could help. "
        "If you have more questions about your health, feel free to ask anytime. "
        "Take care and stay healthy! 💪"
    ),
    "bye": (
        "Goodbye! 👋 Take care of yourself. "
        "Remember, if your symptoms persist, please visit a healthcare professional. "
        "I'm always here whenever you need me. Stay healthy! 🌟"
    ),
    "who": (
        "I'm MedAI 🩺 — an AI-powered health assistant. "
        "I can help analyze your symptoms and provide preliminary health guidance. "
        "Just describe what you're feeling and I'll do my best to help!"
    ),
    "howru": (
        "I'm doing great, thank you for asking! 😊 "
        "More importantly — how are YOU doing? "
        "If you're experiencing any health concerns, feel free to describe your symptoms."
    ),
}

# Medical keywords — if ANY appear, the input is likely medical, not a greeting
_MEDICAL_SIGNALS = {
    "pain", "ache", "fever", "cough", "cold", "nausea", "vomit", "diarrhea",
    "fatigue", "weakness", "swelling", "rash", "itch", "dizzy", "headache",
    "breathless", "palpitation", "numb", "tingling", "burning", "cramp",
    "bleeding", "discharge", "infection", "diabetes", "pressure", "cholesterol",
    "sugar", "thyroid", "urine", "chest", "stomach", "abdomen", "joint",
    "throat", "lung", "kidney", "liver", "heart", "skin", "eye", "ear",
    "pregnant", "period", "weight", "appetite", "vomiting", "symptom",
    "diagnos", "medication", "tablet", "injection", "blood", "report",
}


def detect_greeting(text: str) -> dict | None:
    """
    Check if user input is a greeting or non-medical conversational message.
    Returns a greeting response dict if detected, otherwise None.
    """
    cleaned = text.strip()

    # Skip if text is long (likely medical content even if starts with greeting)
    if len(cleaned.split()) > 15:
        return None

    # If the text contains medical keywords, it's NOT a greeting
    lower = cleaned.lower()
    if any(kw in lower for kw in _MEDICAL_SIGNALS):
        return None

    for pattern in _GREETING_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            # Determine which response category
            if any(w in lower for w in ["thank", "thanks", "thx", "ty"]):
                msg = _GREETING_RESPONSES["thanks"]
            elif any(w in lower for w in ["bye", "goodbye", "see you", "take care"]):
                msg = _GREETING_RESPONSES["bye"]
            elif any(w in lower for w in ["who are you", "what are you", "your name",
                                           "about yourself"]):
                msg = _GREETING_RESPONSES["who"]
            elif any(w in lower for w in ["how are you", "what's up", "wassup", "sup"]):
                msg = _GREETING_RESPONSES["howru"]
            elif any(w in lower for w in ["what can you", "how can you", "what do you do",
                                           "help me", "help", "can you help",
                                           "capabilit", "feature", "function",
                                           "services", "how do you work",
                                           "i need help"]):
                msg = _GREETING_RESPONSES["capability"]
            else:
                msg = _GREETING_RESPONSES["greeting"]

            return {
                "greeting": True,
                "message": msg,
            }

    return None


def _extract_and_combine(request, serializer):
    """
    Common helper: extract symptoms text + optional uploaded file,
    sanitize, and return the combined cleaned text.
    Returns (cleaned_text, file_name, file_text, error_response).
    error_response is None on success.
    """
    symptoms_text = serializer.validated_data.get("symptoms", "").strip()
    report_file = serializer.validated_data.get("report")

    file_text = ""
    file_name = None
    if report_file:
        try:
            file_text = extract_text_from_file(report_file)
            file_name = report_file.name
            logger.info("Extracted %d chars from uploaded file: %s", len(file_text), file_name)
        except ValueError as exc:
            return None, None, None, Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    combined_parts = []
    if symptoms_text:
        combined_parts.append(symptoms_text)
    if file_text:
        combined_parts.append(file_text)
    combined_text = "\n\n".join(combined_parts)

    try:
        cleaned_text = sanitize_input(combined_text)
    except ValueError as exc:
        return None, None, None, Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    return cleaned_text, file_name, file_text, None


@api_view(["POST"])
@parser_classes([JSONParser, MultiPartParser, FormParser])
def predict_view(request):
    """
    POST /api/predict/
    ClinicalBERT classification only.
    Accepts symptom text and/or an uploaded medical report file.
    """
    serializer = SymptomInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {"error": "Invalid input", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    cleaned_text, file_name, file_text, err = _extract_and_combine(request, serializer)
    if err:
        return err

    # Check for greetings / non-medical input before running ML pipeline
    if not file_name:
        greeting = detect_greeting(cleaned_text)
        if greeting:
            return Response(greeting, status=status.HTTP_200_OK)

    try:
        result = predict(cleaned_text)
    except RuntimeError as exc:
        logger.error("Prediction runtime error: %s", exc)
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except Exception:
        logger.exception("Unexpected prediction failure")
        return Response(
            {"error": "An unexpected error occurred during prediction."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if file_name:
        result["uploaded_file"] = file_name

    # Log to database
    top = result["predictions"][0]
    is_emergency = result.get("emergency") is not None
    try:
        PredictionLog.objects.create(
            symptoms=cleaned_text[:2000],
            predicted_disease=top["disease"],
            confidence=top["confidence"],
            risk_level=top["risk_level"],
            is_emergency=is_emergency,
        )
    except Exception:
        logger.exception("Failed to log prediction to database")

    return Response(result, status=status.HTTP_200_OK)


@api_view(["POST"])
@parser_classes([JSONParser, MultiPartParser, FormParser])
def predict_rag_view(request):
    """
    POST /api/predict-rag/
    Full Hybrid RAG Pipeline OR Medical Report Summarization.

    If user uploads a file and asks to "summarise", the report is summarised.
    Otherwise runs: ClinicalBERT → Symptom Verification → FAISS → FLAN-T5.
    """
    serializer = SymptomInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {"error": "Invalid input", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    cleaned_text, file_name, file_text, err = _extract_and_combine(request, serializer)
    if err:
        return err

    # Check for greetings / non-medical input before running ML pipeline
    if not file_name:
        greeting = detect_greeting(cleaned_text)
        if greeting:
            return Response(greeting, status=status.HTTP_200_OK)

    # --- Report Summarization Mode ---
    symptoms_text = serializer.validated_data.get("symptoms", "").strip()
    if file_name and is_summary_request(symptoms_text, has_file=True):
        try:
            summary_result = summarize_report(file_text, file_name)
            return Response(summary_result, status=status.HTTP_200_OK)
        except Exception:
            logger.exception("Report summarization failed")
            return Response(
                {"error": "Failed to summarise the report. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # --- Disease Prediction Mode ---
    try:
        result = predict_with_rag(cleaned_text)
    except RuntimeError as exc:
        logger.error("RAG prediction runtime error: %s", exc)
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except Exception:
        logger.exception("Unexpected RAG prediction failure")
        return Response(
            {"error": "An unexpected error occurred during prediction."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if file_name:
        result["uploaded_file"] = file_name

    # Log to database
    top = result["predictions"][0]
    is_emergency = result.get("emergency") is not None
    try:
        PredictionLog.objects.create(
            symptoms=cleaned_text[:2000],
            predicted_disease=top["disease"],
            confidence=top["confidence"],
            risk_level=top["risk_level"],
            is_emergency=is_emergency,
        )
    except Exception:
        logger.exception("Failed to log prediction to database")

    return Response(result, status=status.HTTP_200_OK)


@api_view(["GET"])
def health_check(request):
    """
    GET /api/health/
    Returns API operational status, GPU info, and RAG component status.
    """
    gpu_info = get_gpu_info()

    rag_status = {}
    try:
        from ml_engine.rag.rag_loader import get_rag_status
        rag_status = get_rag_status()
    except Exception:
        rag_status = {"error": "RAG status unavailable"}

    from ml_engine.model_loader import get_model_info
    model_info = get_model_info()

    return Response({
        "status": "API running",
        "application": "MedAI",
        "version": "2.0.0",
        "gpu": gpu_info,
        "clinicalbert": model_info,
        "rag": rag_status,
    })


@api_view(["GET"])
def prediction_history(request):
    """
    GET /api/history/
    Returns recent prediction logs (latest 50).
    """
    logs = PredictionLog.objects.all()[:50]
    serializer = PredictionLogSerializer(logs, many=True)
    return Response({"history": serializer.data})


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------

@api_view(["POST"])
@permission_classes([AllowAny])
def register_view(request):
    """
    POST /api/auth/register/
    Create a new user account and return a token.
    """
    serializer = RegisterSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {"error": "Registration failed", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = serializer.save()
    token, _ = Token.objects.get_or_create(user=user)
    profile = UserProfile.objects.get(user=user)

    return Response({
        "message": "Account created successfully!",
        "token": token.key,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
        "profile": {
            "avatar_initial": profile.avatar_initial,
        },
    }, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    """
    POST /api/auth/login/
    Authenticate user and return a token.
    """
    serializer = LoginSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {"error": "Invalid input", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = authenticate(
        username=serializer.validated_data["username"],
        password=serializer.validated_data["password"],
    )

    if user is None:
        return Response(
            {"error": "Invalid username or password."},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    if not user.is_active:
        return Response(
            {"error": "This account has been deactivated."},
            status=status.HTTP_403_FORBIDDEN,
        )

    token, _ = Token.objects.get_or_create(user=user)
    profile, _ = UserProfile.objects.get_or_create(user=user)

    return Response({
        "message": "Login successful!",
        "token": token.key,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
        "profile": {
            "avatar_initial": profile.avatar_initial,
            "phone": profile.phone,
            "gender": profile.gender,
            "blood_group": profile.blood_group,
        },
    }, status=status.HTTP_200_OK)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    POST /api/auth/logout/
    Delete user's auth token.
    """
    try:
        request.user.auth_token.delete()
    except Exception:
        pass
    return Response({"message": "Logged out successfully."}, status=status.HTTP_200_OK)


@api_view(["GET", "PUT"])
@permission_classes([IsAuthenticated])
def profile_view(request):
    """
    GET  /api/auth/profile/  — Retrieve current user profile.
    PUT  /api/auth/profile/  — Update current user profile.
    """
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)

    if request.method == "GET":
        serializer = UserProfileSerializer(profile)
        return Response(serializer.data, status=status.HTTP_200_OK)

    # PUT — Update
    serializer = UserProfileUpdateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {"error": "Invalid data", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    data = serializer.validated_data

    # Update User fields
    user_fields = ["first_name", "last_name", "email"]
    for field in user_fields:
        if field in data:
            setattr(user, field, data[field])
    user.save()

    # Update Profile fields
    profile_fields = [
        "phone", "date_of_birth", "gender", "blood_group",
        "height_cm", "weight_kg", "allergies", "medical_conditions",
        "emergency_contact", "address",
    ]
    for field in profile_fields:
        if field in data:
            setattr(profile, field, data[field])
    profile.save()

    return Response(
        UserProfileSerializer(profile).data,
        status=status.HTTP_200_OK,
    )