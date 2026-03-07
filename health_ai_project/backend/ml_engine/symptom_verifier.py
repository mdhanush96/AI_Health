"""
MedAI – Symptom Verification Layer
Compares user-reported symptoms against the knowledge base symptom data
for each ClinicalBERT prediction to re-rank and select the best match.

Pipeline position:
    User Symptoms → ClinicalBERT → Top-3 → **Symptom Verification** → Best Match → KB → T5
"""

import logging
import re

from ml_engine.knowledge_base import get_disease_sections

logger = logging.getLogger("ml_engine.symptom_verifier")

# ---------------------------------------------------------------------------
# Symptom tokenisation helpers
# ---------------------------------------------------------------------------

# Common medical stop-words to ignore during matching
_STOP_WORDS = {
    "i", "have", "had", "am", "is", "are", "was", "were", "been", "be",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "from", "my", "me", "very", "really", "also", "too",
    "some", "like", "feel", "feeling", "getting", "got", "lot", "lots",
    "since", "last", "few", "days", "day", "week", "weeks", "month",
    "experiencing", "experience", "suffering", "noticed", "symptoms",
}


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens, removing stop words."""
    words = set(re.findall(r'[a-z]+', text.lower()))
    return words - _STOP_WORDS


def _extract_symptom_phrases(text: str) -> list[str]:
    """
    Extract meaningful symptom phrases from comma/semicolon separated text.
    Works for both user input and KB symptom strings.
    """
    # Split on commas, semicolons, "and", periods
    parts = re.split(r'[,;.]|\band\b', text.lower())
    phrases = []
    for p in parts:
        p = p.strip()
        if p and len(p) > 2:
            phrases.append(p)
    return phrases


# Synonym groups: any term in a group is treated as equivalent
_SYMPTOM_SYNONYMS = {
    "acid reflux": ["heartburn", "acid reflux", "reflux", "regurgitation", "gerd"],
    "heartburn": ["heartburn", "acid reflux", "reflux", "burning chest"],
    "chest pain": ["chest pain", "chest tightness", "chest discomfort"],
    "headache": ["headache", "head pain", "cephalalgia"],
    "stomach pain": ["stomach pain", "abdominal pain", "epigastric pain", "belly pain"],
    "joint pain": ["joint pain", "arthralgia", "joint aches"],
    "muscle pain": ["muscle pain", "myalgia", "muscle aches", "body aches"],
    "rash": ["rash", "skin rash", "eruption", "red patches", "spots"],
    "itching": ["itching", "pruritus", "itchy", "itch"],
    "fever": ["fever", "high fever", "pyrexia", "temperature", "febrile"],
    "nausea": ["nausea", "nauseous", "feeling sick", "queasy"],
    "vomiting": ["vomiting", "throwing up", "emesis"],
    "diarrhea": ["diarrhea", "loose stools", "watery stools"],
    "cough": ["cough", "coughing", "dry cough", "productive cough"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "weakness", "lethargy"],
    "shortness of breath": ["shortness of breath", "breathlessness", "dyspnea", "breathing difficulty"],
    "swelling": ["swelling", "edema", "swollen", "puffiness"],
    "numbness": ["numbness", "tingling", "paresthesia", "pins and needles"],
    "burning urination": ["burning urination", "painful urination", "dysuria"],
    "frequent urination": ["frequent urination", "polyuria", "urinary frequency"],
    "stiffness": ["stiffness", "rigidity", "limited motion", "reduced range"],
    "wheezing": ["wheezing", "whistling breath", "bronchospasm"],
    "sore throat": ["sore throat", "throat pain", "pharyngitis"],
    "runny nose": ["runny nose", "nasal discharge", "rhinorrhea", "nasal drip"],
    "sneezing": ["sneezing", "sneeze", "sternutation"],
    "weight loss": ["weight loss", "losing weight", "unintended weight loss"],
    "blurred vision": ["blurred vision", "vision blurring", "visual disturbance"],
    "dark urine": ["dark urine", "brown urine", "tea-colored urine"],
    "yellow skin": ["yellow skin", "jaundice", "yellowing", "icterus"],
    "bleeding": ["bleeding", "hemorrhage", "blood loss"],
    "chills": ["chills", "rigors", "shivering"],
}


def _expand_with_synonyms(text: str) -> str:
    """Expand user text by appending synonyms of detected symptoms."""
    text_lower = text.lower()
    expansions = []
    for key, synonyms in _SYMPTOM_SYNONYMS.items():
        if key in text_lower:
            expansions.extend(synonyms)
    if expansions:
        return text_lower + " " + " ".join(expansions)
    return text_lower


def _phrase_overlap_score(user_text: str, kb_symptoms: str) -> float:
    """
    Calculate a symptom match score between user text and KB symptoms.

    Uses a combination of:
    1. Word-level overlap (with synonym expansion)
    2. Phrase-level partial matching (substring containment)
    3. Direct key symptom substring matching

    Returns a score between 0.0 and 1.0.
    """
    if not kb_symptoms or not user_text:
        return 0.0

    # Expand user text with synonyms for better matching
    user_expanded = _expand_with_synonyms(user_text)
    kb_lower = kb_symptoms.lower()

    # --- Word-level overlap (with expanded input) ---
    user_words = _tokenize(user_expanded)
    kb_words = _tokenize(kb_symptoms)

    if not user_words or not kb_words:
        return 0.0

    common_words = user_words & kb_words
    # Jaccard-style: intersection over the smaller set
    word_score = len(common_words) / max(min(len(user_words), len(kb_words)), 1)

    # --- Phrase-level matching ---
    kb_phrases = _extract_symptom_phrases(kb_symptoms)

    phrase_hits = 0
    for kb_phrase in kb_phrases:
        kb_key_words = _tokenize(kb_phrase)
        if not kb_key_words:
            continue
        match_ratio = len(kb_key_words & user_words) / len(kb_key_words)
        if match_ratio >= 0.4:  # Relaxed: 40% word overlap in phrase
            phrase_hits += 1

    phrase_score = phrase_hits / max(len(kb_phrases), 1)

    # --- Direct symptom keyword matching (with synonyms) ---
    key_symptoms = [
        "fever", "headache", "pain", "cough", "rash", "itching", "nausea",
        "vomiting", "diarrhea", "fatigue", "weakness", "swelling", "burning",
        "bleeding", "stiffness", "wheezing", "heartburn", "acid reflux",
        "chest pain", "joint pain", "muscle pain", "abdominal pain",
        "sore throat", "runny nose", "blurred vision", "weight loss",
        "frequent urination", "dark urine", "yellow skin", "shortness of breath",
        "numbness", "tingling", "cramps", "chills", "sweating",
        "regurgitation", "reflux", "congestion", "sneezing",
    ]
    direct_hits = 0
    for kw in key_symptoms:
        if kw in user_expanded and kw in kb_lower:
            direct_hits += 1

    # Normalise: 1 hit = decent, 2+ = strong
    direct_score = min(direct_hits / 2.0, 1.0)

    # --- Combined score (weighted) ---
    combined = (word_score * 0.25) + (phrase_score * 0.35) + (direct_score * 0.40)
    return round(min(combined, 1.0), 4)


# ---------------------------------------------------------------------------
# Main Verification API
# ---------------------------------------------------------------------------

# Threshold below which we consider no disease a strong match
WEAK_MATCH_THRESHOLD = 0.15


def verify_symptoms(
    user_text: str,
    predictions: list[dict],
) -> dict:
    """
    Verify ClinicalBERT predictions against KB symptoms and re-rank.

    Args:
        user_text: Raw user symptom description.
        predictions: List of ClinicalBERT predictions
                     [{"disease": str, "confidence": float, ...}, ...]

    Returns:
        {
            "best_match": str | None,
            "best_score": float,
            "is_strong_match": bool,
            "verified_predictions": [
                {
                    "disease": str,
                    "confidence": float,
                    "symptom_match_score": float,
                    "combined_score": float,
                    ...original fields...
                }
            ],
            "verification_summary": str,
        }
    """
    if not predictions:
        return {
            "best_match": None,
            "best_score": 0.0,
            "is_strong_match": False,
            "verified_predictions": [],
            "verification_summary": "No predictions to verify.",
        }

    scored: list[dict] = []

    for pred in predictions:
        disease = pred["disease"]
        confidence = pred.get("confidence", 0.0)

        # Get KB symptoms for this disease
        kb = get_disease_sections(disease)
        kb_symptoms = kb.get("symptoms", "")

        # Calculate symptom match score
        symptom_score = _phrase_overlap_score(user_text, kb_symptoms)

        # Combined score: ClinicalBERT confidence (normalised 0-1) + symptom match
        # Weight: 60% model confidence, 40% symptom verification
        confidence_norm = confidence / 100.0
        combined = (confidence_norm * 0.6) + (symptom_score * 0.4)

        entry = {
            **pred,
            "symptom_match_score": round(symptom_score, 4),
            "combined_score": round(combined, 4),
        }
        scored.append(entry)

    # Sort by combined score (highest first)
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    best = scored[0]
    best_match = best["disease"]
    best_score = best["combined_score"]
    is_strong = best["symptom_match_score"] >= WEAK_MATCH_THRESHOLD

    # Build summary
    if is_strong:
        summary = (
            f"The symptoms most closely match {best_match.title()} "
            f"(symptom match: {best['symptom_match_score']:.0%}, "
            f"model confidence: {best['confidence']:.1f}%)."
        )
    else:
        disease_names = ", ".join(p["disease"].title() for p in scored[:3])
        summary = (
            f"No single disease is a strong symptom match. "
            f"Possible conditions: {disease_names}. "
            f"Please consult a healthcare professional for accurate diagnosis."
        )

    logger.info(
        "Symptom verification | Best: %s (combined=%.3f, symptom=%.3f) | Strong: %s",
        best_match, best_score, best["symptom_match_score"], is_strong,
    )

    return {
        "best_match": best_match,
        "best_score": best_score,
        "is_strong_match": is_strong,
        "verified_predictions": scored,
        "verification_summary": summary,
    }
