"""
MedAI – Production Predictor Module
ClinicalBERT classification + Hybrid RAG pipeline.
GPU-accelerated inference with risk classification and emergency detection.
"""

import json
import logging
import os
import time

import torch
import torch.nn.functional as F

from . import config
from .model_loader import load_model_and_tokenizer

logger = logging.getLogger("ml_engine")

# ---------------------------------------------------------------------------
# Disease Knowledge Base (static JSON + JSON knowledge base)
# ---------------------------------------------------------------------------
with open(config.DISEASE_INFO_PATH, "r", encoding="utf-8") as _f:
    disease_info = json.load(_f)

# Enrich with per-disease JSON knowledge base
from .knowledge_base import get_disease_sections as _get_kb_sections

# ---------------------------------------------------------------------------
# Emergency Keywords
# ---------------------------------------------------------------------------
EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "unconscious",
    "seizure",
    "severe bleeding",
    "heart attack",
    "stroke",
    "not breathing",
    "choking",
    "suicidal",
]

# Keyword-specific explanations — conversational & caring
_EMERGENCY_EXPLANATIONS = {
    "chest pain": (
        "Chest pain can sometimes be a sign of a serious heart condition. "
        "It's not always something to worry about, but when it comes to "
        "your heart, it's always better to be safe and get checked right away."
    ),
    "heart attack": (
        "Symptoms of a heart attack — such as chest tightness, pain radiating "
        "to the arm or jaw, shortness of breath, or cold sweats — are not "
        "something to take lightly. These need immediate medical attention."
    ),
    "difficulty breathing": (
        "Difficulty breathing can be caused by many things, but sudden or severe "
        "breathlessness needs urgent care to rule out anything serious."
    ),
    "stroke": (
        "Stroke symptoms — sudden numbness, confusion, trouble speaking, or "
        "severe headache — require immediate action. Every minute counts."
    ),
    "unconscious": (
        "Loss of consciousness is a medical emergency. The person needs "
        "immediate professional help."
    ),
    "seizure": (
        "Seizures can look frightening and may indicate an underlying condition "
        "that needs urgent evaluation by a doctor."
    ),
    "severe bleeding": (
        "Severe or uncontrollable bleeding needs immediate medical intervention "
        "to prevent serious complications."
    ),
    "not breathing": (
        "If someone has stopped breathing, this is a life-threatening emergency. "
        "Please call for help immediately and begin CPR if you know how."
    ),
    "choking": (
        "Choking can block the airway and become life-threatening very quickly. "
        "Please seek help right away."
    ),
    "suicidal": (
        "I hear you, and I want you to know that help is available. "
        "You don't have to go through this alone — please reach out to "
        "a counselor or call a helpline right now. Your life matters."
    ),
}


def classify_risk(confidence: float) -> str:
    """Classify prediction confidence into risk levels."""
    if confidence >= 70.0:
        return "High Probability"
    elif confidence >= 40.0:
        return "Moderate Probability"
    else:
        return "Low Confidence"


def detect_emergency(text: str) -> dict | None:
    """Check if user input contains emergency keywords."""
    text_lower = text.lower()
    triggered = [kw for kw in EMERGENCY_KEYWORDS if kw in text_lower]
    if triggered:
        # Build a conversational explanation for the triggered keywords
        explanations = []
        for kw in triggered:
            exp = _EMERGENCY_EXPLANATIONS.get(kw)
            if exp and exp not in explanations:
                explanations.append(exp)
        explanation_text = " ".join(explanations) if explanations else (
            "The symptoms you described may need urgent medical attention."
        )

        return {
            "is_emergency": True,
            "triggered_keywords": triggered,
            "explanation": explanation_text,
            "helplines": (
                "📞 Emergency Helplines:\n"
                "• 112 — Emergency Number (Police, Fire, Ambulance)\n"
                "• 108 — Ambulance / Emergency Medical Services\n"
                "• 102 — Women & Children Ambulance\n"
                "• 1800-599-0019 — Mental Health Helpline (iCall)\n"
                "• 9152987821 — Vandrevala Foundation (24/7 Mental Health)"
            ),
            "message": (
                "Please visit your nearest emergency room or call 112 for immediate help."
            ),
        }
    return None


def predict(text: str, top_k: int = 3) -> dict:
    """
    Run ClinicalBERT inference on symptom text.

    Returns dict with:
        - predictions: list of top-k disease predictions
        - emergency: emergency info if triggered
        - disclaimer: mandatory medical disclaimer
    """
    if not text or not text.strip():
        raise ValueError("Symptom text cannot be empty.")

    model, tokenizer, device = load_model_and_tokenizer()

    # Check for emergencies
    emergency = detect_emergency(text)

    # Tokenize input and move to device (GPU/CPU)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.CLINICALBERT_MAX_LENGTH,
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Inference with gradient disabled for performance
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)

    top_probs, top_ids = torch.topk(probabilities, min(top_k, probabilities.size(1)))

    predictions = []
    for prob, idx in zip(top_probs[0], top_ids[0]):
        disease_name = model.config.id2label[idx.item()]
        confidence = round(prob.item() * 100, 2)
        risk_level = classify_risk(confidence)

        # Merge static disease_info with JSON knowledge base
        info = dict(disease_info.get(disease_name, {}))
        kb = _get_kb_sections(disease_name)
        if kb:
            info["knowledge_base"] = kb
            # Enrich basic fields if missing in disease_info.json
            if "description" not in info and "overview" in kb:
                info["description"] = kb["overview"][:300]

        predictions.append({
            "disease": disease_name,
            "confidence": confidence,
            "risk_level": risk_level,
            "info": info,
        })

    if config.LOG_INFERENCE_INPUTS:
        logger.info(
            "Classification complete | Input: '%s' | Top: %s (%.2f%%) | Device: %s",
            text[:80], predictions[0]["disease"], predictions[0]["confidence"], device,
        )

    return {
        "predictions": predictions,
        "emergency": emergency,
        "disclaimer": (
            "This prediction is generated by an AI model and is NOT a medical diagnosis. "
            "Always consult a qualified healthcare professional for medical advice."
        ),
    }


def predict_with_rag(text: str, top_k: int = 3) -> dict:
    """
    Full Hybrid pipeline:
        User Symptoms → ClinicalBERT → Top-3 → Symptom Verification
        → Best Match → Knowledge Retrieval → FLAN-T5 Generation

    Returns combined result with classification, verification, and RAG response.
    """
    t0 = time.perf_counter()

    # Step 1: ClinicalBERT classification → top-3 predictions
    classification_result = predict(text, top_k=top_k)

    # Step 2: Symptom Verification Layer — re-rank using KB symptoms
    from .symptom_verifier import verify_symptoms
    verification = verify_symptoms(text, classification_result["predictions"])
    verified_predictions = verification["verified_predictions"]

    # The best matching disease after verification (may differ from ClinicalBERT top-1)
    best_disease = verification["best_match"]

    # Step 3: RAG pipeline (retrieval + generation) using verified best match
    rag_result = {}
    if config.RAG_ENABLED:
        try:
            from .rag import rag_query
            rag_result = rag_query(
                query=text,
                classified_disease=best_disease,
                all_predictions=verified_predictions,
            )
        except Exception as exc:
            logger.exception("RAG pipeline failed, returning classification only: %s", exc)
            rag_result = {
                "rag_response": "RAG generation unavailable. Showing classification results only.",
                "retrieved_chunks": [],
                "generation_meta": {},
                "pipeline_latency_ms": 0,
                "cache_hit": False,
                "rag_error": str(exc),
            }

    total_ms = (time.perf_counter() - t0) * 1000

    return {
        **classification_result,
        "predictions": verified_predictions,  # re-ranked predictions
        "symptom_verification": {
            "best_match": verification["best_match"],
            "best_score": verification["best_score"],
            "is_strong_match": verification["is_strong_match"],
            "summary": verification["verification_summary"],
        },
        "rag": rag_result,
        "total_latency_ms": round(total_ms, 1),
    }