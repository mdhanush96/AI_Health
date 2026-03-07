"""
MedAI – Medical Named Entity Recognition (NER)
Uses SciSpacy (en_core_sci_md) to automatically detect medical entities
from report text — diseases, medications, symptoms, procedures, tests, etc.

This replaces hardcoded regex lists with an ML-based approach that can
recognize thousands of medical terms without manual curation.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("ml_engine")

# ---------------------------------------------------------------------------
# Lazy-loaded SciSpacy model (cached after first load)
# ---------------------------------------------------------------------------

_nlp = None
_NER_AVAILABLE = None  # None = not checked yet


def _load_ner_model():
    """Load the SciSpacy biomedical NER model (lazy, cached)."""
    global _nlp, _NER_AVAILABLE
    if _nlp is not None:
        return _nlp

    try:
        import spacy

        t0 = time.perf_counter()
        _nlp = spacy.load("en_core_sci_md")
        elapsed = time.perf_counter() - t0
        _NER_AVAILABLE = True
        logger.info("SciSpacy en_core_sci_md loaded | %.2fs", elapsed)
        return _nlp
    except Exception as exc:
        _NER_AVAILABLE = False
        logger.warning("SciSpacy NER not available: %s", exc)
        return None


def is_ner_available() -> bool:
    """Check if the NER model can be loaded."""
    global _NER_AVAILABLE
    if _NER_AVAILABLE is None:
        _load_ner_model()
    return bool(_NER_AVAILABLE)


# ---------------------------------------------------------------------------
# Entity classification helpers
# ---------------------------------------------------------------------------

# SciSpacy en_core_sci_md labels all entities as "ENTITY" — we use UMLS
# semantic types (if linked) or heuristics to classify them.
# These keyword sets help classify unlabeled biomedical entities.

_MEDICATION_INDICATORS = {
    "mg", "mcg", "ml", "tablet", "capsule", "injection", "syrup",
    "twice", "daily", "once", "thrice", "bid", "tid", "od", "hs",
    "oral", "iv", "im", "sc", "topical", "inhaler", "drops",
}

_KNOWN_DRUG_SUFFIXES = [
    "statin", "pril", "sartan", "olol", "pine", "azole", "mycin",
    "cillin", "cycline", "prazole", "gliptin", "flozin", "mab",
    "tinib", "parin", "formin", "pressin", "navir", "vudine",
    "zosin", "dipine", "lukast", "fenac", "profen", "setron",
    "tropin", "taxel", "platin",
]

_SYMPTOM_KEYWORDS = {
    "pain", "ache", "fever", "cough", "nausea", "vomiting", "diarrhea",
    "fatigue", "weakness", "swelling", "bleeding", "itching", "rash",
    "dizziness", "headache", "breathlessness", "dyspnea", "palpitation",
    "numbness", "tingling", "burning", "cramp", "stiffness", "soreness",
    "loss of appetite", "weight loss", "weight gain", "insomnia",
    "constipation", "bloating", "chest pain", "abdominal pain",
    "joint pain", "back pain", "urinary", "discharge",
}

_DISEASE_KEYWORDS = {
    "disease", "disorder", "syndrome", "infection", "itis",
    "osis", "emia", "opathy", "oma", "algia", "ectomy",
    "diabetes", "hypertension", "cancer", "tumor", "carcinoma",
    "failure", "insufficiency", "deficiency", "anemia", "anaemia",
    "hepatitis", "cirrhosis", "pneumonia", "bronchitis", "asthma",
    "arthritis", "neuropathy", "retinopathy", "nephropathy",
    "stroke", "infarction", "thrombosis", "embolism",
}

_TEST_KEYWORDS = {
    "test", "assay", "level", "count", "ratio", "index", "panel",
    "culture", "biopsy", "scan", "x-ray", "xray", "mri", "ct",
    "ultrasound", "ecg", "ekg", "echocardiogram", "endoscopy",
    "colonoscopy", "mammography", "hemoglobin", "hba1c", "creatinine",
    "bilirubin", "albumin", "cholesterol", "triglyceride", "glucose",
    "urea", "electrolyte", "cbc", "lft", "rft", "kft", "tft",
}

_PROCEDURE_KEYWORDS = {
    "surgery", "operation", "procedure", "transplant", "bypass",
    "angioplasty", "stent", "catheter", "dialysis", "amputation",
    "biopsy", "excision", "debridement", "drainage", "intubation",
    "ventilation", "transfusion", "chemotherapy", "radiation",
}

_ANATOMY_KEYWORDS = {
    "liver", "kidney", "heart", "lung", "brain", "pancreas",
    "thyroid", "spleen", "stomach", "intestine", "colon", "bladder",
    "uterus", "ovary", "prostate", "breast", "bone", "joint",
    "artery", "vein", "nerve", "muscle", "skin", "eye", "ear",
}

# Entities to filter out (too generic or not medically useful)
_NOISE_ENTITIES = {
    # People / demographics
    "patient", "doctor", "physician", "nurse", "surgeon", "specialist",
    "male", "female", "age", "year", "month", "day", "date", "time",
    "mr", "mrs", "ms", "dr", "name", "address", "phone", "email",
    "male patient", "female patient", "patient demographics",
    # Document structure
    "report", "hospital", "clinic", "institute", "medical center",
    "laboratory", "department", "unit",
    "history", "examination", "finding", "result", "value", "range",
    "normal", "abnormal", "positive", "negative", "present", "absent",
    # Severity / generic descriptors
    "mild", "moderate", "severe", "chronic", "acute", "stable",
    "follow", "primary", "secondary", "initial", "final",
    "twice daily", "once daily", "oral", "mg", "ml",
    # Generic clinical phrases that are NOT diagnoses
    "health status", "disease progression", "adjusting treatment",
    "smoking status", "alcohol consumption", "diet preference",
    "exercise habits", "primary diagnosis", "patient vitals",
    "oxygen saturation", "blood pressure", "heart rate",
    "respiratory rate", "temperature", "body weight", "body height",
    "clinical summary", "medical history", "family history",
    "social history", "review of systems", "physical examination",
    "vital signs", "general appearance", "treatment plan",
    "follow up", "follow-up", "chief complaint",
    "presenting complaint", "reason for visit",
}

# Keywords indicating an institution / org (not a medical entity)
_INSTITUTION_KEYWORDS = {
    "institute", "inc", "llc", "ltd", "hospital", "clinic", "center",
    "centre", "medical group", "health system", "healthcare",
    "foundation", "university", "college", "laboratory", "labs",
    "associates", "partners", "practice", "department",
}


def _looks_like_person_name(text: str) -> bool:
    """Heuristic: detect probable person names (e.g. 'James Campbell')."""
    words = text.split()
    if len(words) < 2 or len(words) > 4:
        return False
    # If all words are title-case and none match medical keywords → name
    if all(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1):
        combined = text.lower()
        for kw_set in (_DISEASE_KEYWORDS, _SYMPTOM_KEYWORDS, _TEST_KEYWORDS,
                       _PROCEDURE_KEYWORDS, _MEDICATION_INDICATORS):
            for kw in kw_set:
                if kw in combined:
                    return False
        return True
    return False


def _looks_like_institution(text: str) -> bool:
    """Heuristic: detect hospital / company names."""
    t = text.lower()
    return any(kw in t for kw in _INSTITUTION_KEYWORDS)


def _classify_entity(text: str, context: str = "") -> Optional[str]:
    """
    Classify a SciSpacy entity into a medical category.
    Returns: 'disease', 'medication', 'symptom', 'test', 'procedure',
             'anatomy', or None (filtered out).
    """
    t = text.lower().strip()

    # ── Hard filters ──
    if t in _NOISE_ENTITIES or len(t) < 3:
        return None

    # Filter pure numbers / units
    if t.replace(".", "").replace(",", "").replace("/", "").isdigit():
        return None
    # Filter bare units like "mmHg", "mg/dL"
    if t in {"mmhg", "mg/dl", "g/dl", "u/l", "bpm", "kg", "cm", "ml", "mg",
             "iu/l", "meq/l", "mmol/l", "ng/ml", "pg/ml"}:
        return None

    # Filter person names
    if _looks_like_person_name(text):
        return None

    # Filter institution / org names
    if _looks_like_institution(text):
        return None

    # ── Positive classification ──

    # Check drug suffixes first (very reliable signal)
    for suffix in _KNOWN_DRUG_SUFFIXES:
        if t.endswith(suffix):
            return "medication"

    # Check context for medication clues ONLY if the entity itself
    # looks like a drug name (not a section header or descriptor)
    ctx = context.lower()
    if any(ind in ctx for ind in _MEDICATION_INDICATORS):
        # Only classify as med if entity is a single word or
        # looks like "DrugName dose" pattern
        words = t.split()
        if len(words) <= 3 and not any(w in t for w in {
            "patient", "diagnosis", "primary", "status", "blood",
            "oxygen", "pressure", "saturation", "demographics",
            "female", "male", "lifestyle", "follow",
        }):
            return "medication"

    # Check against keyword sets
    for word in t.split():
        if word in _SYMPTOM_KEYWORDS:
            return "symptom"

    for keyword in _DISEASE_KEYWORDS:
        if keyword in t:
            return "disease"

    for keyword in _TEST_KEYWORDS:
        if keyword in t:
            return "test"

    for keyword in _PROCEDURE_KEYWORDS:
        if keyword in t:
            return "procedure"

    for keyword in _ANATOMY_KEYWORDS:
        if keyword in t:
            return "anatomy"

    # ─ NO fallback – unrecognised entities are dropped ─
    # (Previously: multi-word → disease. This caused names, institutions,
    #  and generic phrases to be misclassified.)
    return None


# ---------------------------------------------------------------------------
# Main NER extraction function
# ---------------------------------------------------------------------------

def extract_medical_entities(text: str) -> dict:
    """
    Extract medical entities from text using SciSpacy NER.

    Args:
        text: Raw or cleaned medical report text.

    Returns:
        dict with keys: diseases, medications, symptoms, tests,
        procedures, anatomy, all_entities.
        Each is a deduplicated list of strings.
        Returns empty categories if NER is not available.
    """
    result = {
        "diseases": [],
        "medications": [],
        "symptoms": [],
        "tests": [],
        "procedures": [],
        "anatomy": [],
        "all_entities": [],
    }

    nlp = _load_ner_model()
    if nlp is None:
        logger.warning("NER extraction skipped — model not available")
        return result

    t0 = time.perf_counter()

    # Process text (limit to ~10k chars to keep NER fast on CPU)
    truncated = text[:10000]
    doc = nlp(truncated)

    # Deduplicate by normalized form
    seen = set()

    for ent in doc.ents:
        ent_text = ent.text.strip()
        normalized = ent_text.lower()

        # Skip duplicates
        if normalized in seen:
            continue

        # Get surrounding context for better classification
        start = max(0, ent.start_char - 30)
        end = min(len(truncated), ent.end_char + 30)
        context = truncated[start:end]

        category = _classify_entity(ent_text, context)
        if category is None:
            continue

        seen.add(normalized)

        # Title-case for display
        display_text = ent_text.title() if ent_text.islower() else ent_text

        if category == "disease":
            result["diseases"].append(display_text)
        elif category == "medication":
            result["medications"].append(display_text)
        elif category == "symptom":
            result["symptoms"].append(display_text)
        elif category == "test":
            result["tests"].append(display_text)
        elif category == "procedure":
            result["procedures"].append(display_text)
        elif category == "anatomy":
            result["anatomy"].append(display_text)

        result["all_entities"].append(f"[{category}] {display_text}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "NER extraction: %d entities | diseases=%d, meds=%d, symptoms=%d, "
        "tests=%d, procedures=%d | %.1fms",
        len(result["all_entities"]),
        len(result["diseases"]),
        len(result["medications"]),
        len(result["symptoms"]),
        len(result["tests"]),
        len(result["procedures"]),
        elapsed_ms,
    )

    return result
