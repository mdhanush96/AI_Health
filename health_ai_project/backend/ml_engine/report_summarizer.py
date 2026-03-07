"""
MedAI – Medical Report Summarizer
Uses multi-pass FLAN-T5 questioning + comprehensive structured text extraction
to produce a detailed, human-centric summary of uploaded medical reports.
"""

import logging
import re
import time

import torch

from ml_engine import config
from ml_engine.medical_ner import extract_medical_entities, is_ner_available, _looks_like_person_name
from ml_engine.rag.rag_loader import load_generator_model

logger = logging.getLogger("ml_engine")

# ---------------------------------------------------------------------------
# Summarization intent detection
# ---------------------------------------------------------------------------

_SUMMARY_PATTERNS = [
    r"\bsummar(ise|ize|y)\b",
    r"\bexplain\s*(this|the|my)?\s*report\b",
    r"\bwhat\s*does\s*(this|the|my)\s*report\s*(say|mean|show)\b",
    r"\bread\s*(this|the|my)\s*report\b",
    r"\bbreak\s*down\s*(this|the|my)?\s*report\b",
    r"\banalyse|analyze\s*(this|the|my)?\s*report\b",
    r"\breview\s*(this|the|my)?\s*report\b",
    r"\bwhat\s*is\s*in\s*(this|the|my)\s*report\b",
    r"\btell\s*me\s*about\s*(this|the|my)\s*report\b",
    r"\bunderstand\s*(this|the|my)\s*report\b",
]


def is_summary_request(user_text: str, has_file: bool) -> bool:
    if not has_file:
        return False
    text_lower = user_text.lower().strip()
    for pattern in _SUMMARY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Report text pre-processing
# ---------------------------------------------------------------------------

def _clean_report_text(raw_text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', raw_text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    return text.strip()


def _merge_unique(regex_list: list[str], ner_list: list[str]) -> list[str]:
    """Merge NER-detected items into regex-extracted list, avoiding duplicates."""
    seen = {item.lower().strip() for item in regex_list}
    merged = list(regex_list)
    for item in ner_list:
        normalized = item.lower().strip()
        # Skip if already present (fuzzy: check substring match too)
        if normalized in seen:
            continue
        if any(normalized in s or s in normalized for s in seen):
            continue
        seen.add(normalized)
        merged.append(item)
    return merged


def _looks_like_noise_ner(text: str) -> bool:
    """Return True if a NER entity is noise (person name, generic label, etc.)."""
    t = text.lower().strip()
    # Person name check
    if _looks_like_person_name(text):
        return True
    # Contains role titles
    if any(role in t for role in ("doctor", "nurse", "physician", "surgeon",
                                   "patient", "technician", "consultant")):
        return True
    # Too short
    if len(t) < 3:
        return True
    return False


# ---------------------------------------------------------------------------
# Multi-pass T5 question prompts
# ---------------------------------------------------------------------------

_FOCUSED_QUESTIONS = [
    (
        "patient",
        "What are the patient details in this medical report? "
        "Include name, age, gender, date of visit, and patient ID.\n\n"
        "Report:\n{text}\n\nPatient details:",
    ),
    (
        "diagnosis",
        "What diagnoses or medical conditions are mentioned in this report?\n\n"
        "Report:\n{text}\n\nDiagnoses:",
    ),
    (
        "findings",
        "What are the key clinical findings and test results in this report? "
        "List each test with its value.\n\n"
        "Report:\n{text}\n\nFindings:",
    ),
    (
        "medications",
        "What medications and treatments are mentioned in this report?\n\n"
        "Report:\n{text}\n\nMedications:",
    ),
    (
        "recommendations",
        "What are the doctor's recommendations and follow-up instructions "
        "in this report?\n\n"
        "Report:\n{text}\n\nRecommendations:",
    ),
]


def _ask_t5(model, tokenizer, device, prompt: str, min_tokens: int = 10) -> str:
    """Run a single T5 inference with a focused prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.GENERATOR_MAX_INPUT_TOKENS,
        padding=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            min_new_tokens=min_tokens,
            no_repeat_ngram_size=4,
            early_stopping=True,
            do_sample=False,
            num_beams=2,
            length_penalty=1.0,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return _clean_t5_output(text)


# ---------------------------------------------------------------------------
# Comprehensive structured extraction (regex-based)
# ---------------------------------------------------------------------------

def _extract_patient_info(text: str) -> dict:
    """Extract patient demographics from report text."""
    info = {}
    t = text

    # Name patterns
    for pat in [
        r"(?:patient\s*(?:name)?|name)\s*[:\-]\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})",
        r"(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2})",
        r"(?:Dear|Report\s+(?:of|for))\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2})",
    ]:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            info["name"] = m.group(1).strip()
            break

    # Age
    m = re.search(r'(?:age|aged)\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?|y/?o)?', t, re.I)
    if m:
        info["age"] = m.group(1)
    else:
        m = re.search(r'(\d{1,3})\s*[-/]?\s*(?:year|yr)\s*(?:old)?', t, re.I)
        if m:
            info["age"] = m.group(1)

    # Gender
    m = re.search(r'(?:gender|sex)\s*[:\-]?\s*(male|female|m|f)\b', t, re.I)
    if m:
        g = m.group(1).strip().lower()
        info["gender"] = "Male" if g in ("m", "male") else "Female"
    else:
        if re.search(r'\b(male|man)\b', t, re.I):
            info["gender"] = "Male"
        elif re.search(r'\b(female|woman)\b', t, re.I):
            info["gender"] = "Female"

    # Date
    for pat in [
        r'(?:date|dated|report\s*date|visit\s*date)\s*[:\-]?\s*([\d]{1,2}[/\-.][\d]{1,2}[/\-.][\d]{2,4})',
        r'(?:date|dated)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})',
        r'(\w+\s+\d{1,2},?\s+\d{4})',
    ]:
        m = re.search(pat, t, re.I)
        if m:
            info["date"] = m.group(1).strip()
            break

    # Patient ID / MRN
    m = re.search(r'(?:patient\s*id|mrn|medical\s*record|record\s*no|uhid)\s*[:\-#]?\s*(\w[\w\-]+)', t, re.I)
    if m:
        info["patient_id"] = m.group(1).strip()

    return info


# Words that should NEVER appear in a diagnosis entry
_DIAGNOSIS_BLACKLIST = {
    "patient demographics", "primary diagnosis", "health status",
    "disease progression", "adjusting treatment", "smoking status",
    "alcohol consumption", "diet preference", "exercise habits",
    "patient vitals", "oxygen saturation", "blood pressure",
    "heart rate", "respiratory rate", "temperature", "female patient",
    "male patient", "treatment plan", "follow up", "follow-up",
    "clinical summary", "vital signs", "general appearance",
    "patient information", "report date", "visit date",
}


def _is_real_diagnosis(item: str) -> bool:
    """Return True only if item looks like a real medical diagnosis."""
    t = item.lower().strip()
    if t in _DIAGNOSIS_BLACKLIST:
        return False
    # Reject items that start with filler words
    if re.match(r'^(of|and|the|a|an|or|in|to|for|with|by|at)\s', t):
        return False
    # Reject items that look like section headers (exactly 1-2 generic words)
    if len(t.split()) <= 2 and t in _DIAGNOSIS_BLACKLIST:
        return False
    # Reject if it's just a person name (title-case words, no medical keywords)
    words = item.split()
    if (len(words) >= 2 and all(
            w[0].isupper() and w[1:].islower() for w in words if len(w) > 1
        ) and not any(
            kw in t for kw in (
                "disease", "disorder", "syndrome", "infection", "diabetes",
                "hypertension", "neuropathy", "failure", "anemia", "cancer",
                "itis", "osis", "emia", "opathy",
            )
        )):
        return False
    # Too short or too long
    if len(t) < 4 or len(t) > 100:
        return False
    return True


def _extract_diagnoses(text: str) -> list[str]:
    """Extract diagnoses and conditions from report text."""
    diagnoses = []
    t_lower = text.lower()

    # Look for explicit diagnosis sections
    diag_section = re.search(
        r'(?:diagnosis|diagnoses|impression|assessment|clinical\s*diagnosis)'
        r'\s*[:\-]?\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)',
        text, re.I | re.S
    )
    if diag_section:
        section = diag_section.group(1).strip()
        items = re.split(r'\n|[•\-\d]+[.)]\s*', section)
        for item in items:
            item = item.strip().rstrip('.')
            if _is_real_diagnosis(item):
                diagnoses.append(item)

    # Common diseases / conditions mentioned anywhere
    conditions = [
        r'type\s*[12]\s*diabetes\s*(?:mellitus)?',
        r'hypertension',
        r'hyperthyroid(?:ism)?', r'hypothyroid(?:ism)?',
        r'anemia|anaemia',
        r'pneumonia', r'bronchitis', r'asthma',
        r'coronary\s*artery\s*disease', r'heart\s*failure',
        r'chronic\s*kidney\s*disease', r'renal\s*failure',
        r'liver\s*(?:disease|cirrhosis|failure)',
        r'hepatitis\s*[a-e]?',
        r'dengue', r'malaria', r'typhoid',
        r'tuberculosis|tb\b',
        r'urinary\s*tract\s*infection|uti\b',
        r'dyslipidemia|hyperlipidemia',
        r'obesity', r'osteoporosis',
        r'arthritis', r'gout',
        r'migraine', r'epilepsy',
        r'depression', r'anxiety\s*disorder',
        r'copd',
        r'peripheral\s*neuropathy',
        r'diabetic\s*(?:neuropathy|retinopathy|nephropathy)',
    ]
    seen = set()
    for cond_pat in conditions:
        m = re.search(cond_pat, t_lower)
        if m:
            matched = m.group(0).strip().title()
            if matched.lower() not in seen:
                seen.add(matched.lower())
                if matched not in diagnoses:
                    diagnoses.append(matched)

    return diagnoses


# Words that should never be a "medication"
_MEDICATION_BLACKLIST_WORDS = {
    "patient", "demographics", "diagnosis", "primary", "status",
    "blood", "pressure", "oxygen", "saturation", "female", "male",
    "lifestyle", "follow", "mmhg", "bpm", "adjustment",
    "essential", "monitoring", "needed", "report",
}


def _is_real_medication(item: str) -> bool:
    """Return True only if item looks like a real medication entry."""
    t = item.lower().strip()
    # Too short or too long
    if len(t) < 3 or len(t) > 80:
        return False
    # Reject full sentences (>10 words)
    if len(t.split()) > 10:
        return False
    # Reject items containing blacklisted words
    for bw in _MEDICATION_BLACKLIST_WORDS:
        if bw in t:
            return False
    # Reject bare units
    if t in {"mg", "ml", "mmhg", "bpm", "u/l", "mg/dl", "g/dl"}:
        return False
    return True


def _extract_medications(text: str) -> list[str]:
    """Extract medications from report text."""
    medications = []

    # Look for medication section
    med_section = re.search(
        r'(?:medications?|prescription|drugs?|medicines?|rx)\s*[:\-]?\s*(.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|$)',
        text, re.I | re.S
    )
    if med_section:
        section = med_section.group(1).strip()
        items = re.split(r'\n|[•\-\d]+[.)]\s*|,\s*', section)
        for item in items:
            item = item.strip().rstrip('.')
            if _is_real_medication(item):
                medications.append(item)

    # Common medication patterns: "Drug Name dose"
    med_patterns = [
        r'((?:metformin|insulin|glimepiride|sitagliptin)\s*[\d]*\s*(?:mg|units?)?)',
        r'((?:amlodipine|losartan|telmisartan|atenolol|enalapril|ramipril)\s*[\d]*\s*(?:mg)?)',
        r'((?:atorvastatin|rosuvastatin|simvastatin)\s*[\d]*\s*(?:mg)?)',
        r'((?:aspirin|clopidogrel|warfarin|enoxaparin)\s*[\d]*\s*(?:mg)?)',
        r'((?:omeprazole|pantoprazole|rabeprazole|esomeprazole)\s*[\d]*\s*(?:mg)?)',
        r'((?:paracetamol|acetaminophen|ibuprofen|diclofenac)\s*[\d]*\s*(?:mg)?)',
        r'((?:amoxicillin|azithromycin|ciprofloxacin|doxycycline|ceftriaxone)\s*[\d]*\s*(?:mg)?)',
        r'((?:levothyroxine|thyroxine)\s*[\d]*\s*(?:mcg|mg)?)',
        r'((?:prednisolone|prednisone|dexamethasone|hydrocortisone)\s*[\d]*\s*(?:mg)?)',
        r'((?:cetirizine|montelukast|salbutamol|budesonide)\s*[\d]*\s*(?:mg)?)',
        r'((?:gabapentin|pregabalin|duloxetine|amitriptyline)\s*[\d]*\s*(?:mg)?)',
    ]
    t_lower = text.lower()
    seen = {m.lower() for m in medications}
    for pat in med_patterns:
        m = re.search(pat, t_lower)
        if m:
            med = m.group(1).strip().title()
            if med.lower() not in seen:
                seen.add(med.lower())
                medications.append(med)

    return medications


def _is_real_recommendation(item: str) -> bool:
    """Return True only if item looks like a real recommendation."""
    t = item.lower().strip()
    # Reject raw key-value pairs like "smoking status: former"
    if re.match(r'^\w[\w\s]{1,25}:\s*\w', t):
        # Allow if it looks like a sentence (has a verb-like structure)
        if not any(v in t for v in ('should', 'advised', 'recommend', 'monitor',
                                     'follow', 'avoid', 'exercise', 'diet',
                                     'manage', 'maintain', 'consult', 'review')):
            return False
    # Reject section headers
    if t in _DIAGNOSIS_BLACKLIST:
        return False
    # Too short or too long
    if len(t) < 8 or len(t) > 250:
        return False
    # Reject if fewer than 3 words (likely a label, not a recommendation)
    if len(t.split()) < 3:
        return False
    return True


def _extract_recommendations(text: str) -> list[str]:
    """Extract doctor recommendations and follow-up instructions."""
    recommendations = []

    # Look for recommendations / advice sections
    rec_section = re.search(
        r'(?:recommendation|advice|follow[\s-]*up|plan|instructions?|counsel(?:l?ing)?)\s*[:\-]?\s*(.+?)(?:\n\n|\n(?=[A-Z][a-z]+:)|$)',
        text, re.I | re.S
    )
    if rec_section:
        section = rec_section.group(1).strip()
        items = re.split(r'\n|[•\-\d]+[.)]\s*', section)
        for item in items:
            item = item.strip().rstrip('.')
            if _is_real_recommendation(item):
                recommendations.append(item)

    # Common recommendation phrases
    rec_phrases = [
        r'(?:follow[\s-]*up|review|revisit)\s+(?:in|after|within)\s+[\d\w\s]+',
        r'(?:advised?|recommended?|suggested?)\s+(?:to\s+)?[^.]+',
        r'(?:monitor|check|watch|observe)\s+[^.]+(?:regularly|daily|weekly)?',
        r'(?:avoid|refrain|abstain|stop|quit)\s+[^.]+',
        r'(?:refer(?:red)?|consult)\s+(?:to\s+)?[^.]+',
    ]
    t_lower = text.lower()
    for pat in rec_phrases:
        matches = re.findall(pat, t_lower)
        for m in matches[:2]:
            m = m.strip().capitalize()
            if len(m) > 8 and m not in recommendations:
                recommendations.append(m)

    return recommendations[:6]


def _extract_lab_values(text: str) -> list[dict]:
    """
    Extract lab test results with values and units.
    Returns list of dicts with label, value, unit, and interpretation.
    """
    findings = []
    patterns = [
        (r'(?:hemoglobin|hb)\s*[:\-]?\s*([\d.]+)\s*(g/dl|gm/dl|g%)?', 'Hemoglobin', 'g/dL'),
        (r'(?:wbc|white\s*blood\s*cell)\s*(?:count)?\s*[:\-]?\s*([\d,.]+)\s*(/cumm|cells?/ul|x10\^?3)?', 'WBC Count', '/cumm'),
        (r'(?:rbc|red\s*blood\s*cell)\s*(?:count)?\s*[:\-]?\s*([\d,.]+)\s*(million/cumm|x10\^?6)?', 'RBC Count', 'million/cumm'),
        (r'(?:platelet)\s*(?:count)?\s*[:\-]?\s*([\d,.]+)\s*(/cumm|x10\^?3)?', 'Platelet Count', '/cumm'),
        (r'(?:blood\s*sugar|glucose|fasting\s*(?:blood\s*)?sugar|fbs|rbs|random\s*(?:blood\s*)?sugar)\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Blood Sugar', 'mg/dL'),
        (r'(?:bp|blood\s*pressure)\s*[:\-]?\s*(\d+\s*/\s*\d+)\s*(mmhg)?', 'Blood Pressure', 'mmHg'),
        (r'(?:temperature|temp)\s*[:\-]?\s*([\d.]+)\s*[°]?\s*([fFcC])?', 'Temperature', '°F'),
        (r'(?:creatinine)\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Creatinine', 'mg/dL'),
        (r'(?:total\s*)?(?:cholesterol)\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Cholesterol', 'mg/dL'),
        (r'(?:total\s*)?(?:bilirubin)\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Bilirubin', 'mg/dL'),
        (r'(?:thyroid|tsh)\s*[:\-]?\s*([\d.]+)\s*(miu/l|uiu/ml)?', 'TSH', 'mIU/L'),
        (r'(?:hba1c|a1c|glycated\s*h(?:ae)?moglobin)\s*[:\-]?\s*([\d.]+)\s*(%)?', 'HbA1c', '%'),
        (r'(?:sgot|ast|aspartate)\s*[:\-]?\s*([\d.]+)\s*(u/l|iu/l)?', 'SGOT/AST', 'U/L'),
        (r'(?:sgpt|alt|alanine)\s*[:\-]?\s*([\d.]+)\s*(u/l|iu/l)?', 'SGPT/ALT', 'U/L'),
        (r'(?:urea|blood\s*urea)\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Blood Urea', 'mg/dL'),
        (r'(?:uric\s*acid)\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Uric Acid', 'mg/dL'),
        (r'(?:sodium|na\+?)\s*[:\-]?\s*([\d.]+)\s*(meq/l|mmol/l)?', 'Sodium', 'mEq/L'),
        (r'(?:potassium|k\+?)\s*[:\-]?\s*([\d.]+)\s*(meq/l|mmol/l)?', 'Potassium', 'mEq/L'),
        (r'(?:calcium|ca\+?)\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Calcium', 'mg/dL'),
        (r'(?:ldl)\s*(?:cholesterol)?\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'LDL Cholesterol', 'mg/dL'),
        (r'(?:hdl)\s*(?:cholesterol)?\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'HDL Cholesterol', 'mg/dL'),
        (r'(?:triglyceride)\s*s?\s*[:\-]?\s*([\d.]+)\s*(mg/dl)?', 'Triglycerides', 'mg/dL'),
        (r'(?:esr|sed\s*rate)\s*[:\-]?\s*([\d.]+)\s*(mm/hr)?', 'ESR', 'mm/hr'),
        (r'(?:albumin)\s*[:\-]?\s*([\d.]+)\s*(g/dl)?', 'Albumin', 'g/dL'),
        (r'(?:total\s*protein)\s*[:\-]?\s*([\d.]+)\s*(g/dl)?', 'Total Protein', 'g/dL'),
        (r'(?:vitamin\s*d|vit\s*d)\s*[:\-]?\s*([\d.]+)\s*(ng/ml)?', 'Vitamin D', 'ng/mL'),
        (r'(?:vitamin\s*b12|vit\s*b12)\s*[:\-]?\s*([\d.]+)\s*(pg/ml)?', 'Vitamin B12', 'pg/mL'),
        (r'(?:iron|serum\s*iron)\s*[:\-]?\s*([\d.]+)\s*(mcg/dl|ug/dl)?', 'Iron', 'mcg/dL'),
        (r'(?:ferritin)\s*[:\-]?\s*([\d.]+)\s*(ng/ml)?', 'Ferritin', 'ng/mL'),
        (r'(?:pulse|heart\s*rate|hr)\s*[:\-]?\s*(\d{2,3})\s*(?:bpm|/min)?', 'Heart Rate', 'bpm'),
        (r'(?:spo2|oxygen\s*saturation|o2\s*sat)\s*[:\-]?\s*(\d{2,3})\s*(%)?', 'SpO2', '%'),
        (r'(?:respiratory\s*rate|rr)\s*[:\-]?\s*(\d{1,2})\s*(?:/min|breaths)?', 'Respiratory Rate', '/min'),
        (r'(?:bmi|body\s*mass\s*index)\s*[:\-]?\s*([\d.]+)', 'BMI', 'kg/m²'),
        (r'(?:weight|wt)\s*[:\-]?\s*([\d.]+)\s*(kg|lbs?)?', 'Weight', 'kg'),
        (r'(?:height|ht)\s*[:\-]?\s*([\d.]+)\s*(cm|m|ft|feet)?', 'Height', 'cm'),
    ]

    text_lower = text.lower()
    seen = set()

    for pattern, label, default_unit in patterns:
        match = re.search(pattern, text_lower)
        if match and label not in seen:
            value = match.group(1).strip()

            # ── Validate value ──
            # Skip if value is empty, just ".", or has no actual digits
            clean_val = value.replace(".", "").replace(",", "").strip()
            if not clean_val or not any(c.isdigit() for c in clean_val):
                continue

            # Skip Vitamin D if value looks like a dosage (e.g. 2000 IU)
            if label == "Vitamin D":
                try:
                    vd_val = float(value)
                    # Normal range 20-100 ng/mL; >500 is almost certainly IU dosage
                    if vd_val > 500:
                        continue
                except ValueError:
                    pass

            unit = ""
            if match.lastindex >= 2 and match.group(2):
                unit = match.group(2).strip()
            if not unit:
                unit = default_unit
            note = _get_value_note(label, value)
            findings.append({
                "label": label,
                "value": value,
                "unit": unit,
                "note": note,
            })
            seen.add(label)

    return findings


# ---------------------------------------------------------------------------
# Main summarizer
# ---------------------------------------------------------------------------

def summarize_report(report_text: str, file_name: str = None) -> dict:
    """
    Summarize a medical report using multi-pass T5 + structured extraction.
    """
    t0 = time.perf_counter()

    cleaned_text = _clean_report_text(report_text)

    if len(cleaned_text) < 20:
        return {
            "summary_type": "report_summary",
            "file_name": file_name,
            "summary": (
                "The uploaded file doesn't seem to contain enough text to summarise. "
                "Please upload a clearer report or describe what you'd like to know."
            ),
            "report_length": len(cleaned_text),
            "latency_ms": 0,
        }

    # ── Phase 1a: Medical NER extraction (SciSpacy) ──
    ner_entities = extract_medical_entities(cleaned_text)

    # ── Phase 1b: Structured regex extraction (fast, reliable) ──
    patient_info = _extract_patient_info(cleaned_text)
    diagnoses = _extract_diagnoses(cleaned_text)
    medications = _extract_medications(cleaned_text)
    recommendations = _extract_recommendations(cleaned_text)
    lab_values = _extract_lab_values(cleaned_text)

    # ── Merge NER results with regex results (NER fills gaps) ──
    diagnoses = _merge_unique(diagnoses, ner_entities.get("diseases", []))
    medications = _merge_unique(medications, ner_entities.get("medications", []))

    # Post-merge cleanup: filter out any remaining noise
    diagnoses = [d for d in diagnoses if _is_real_diagnosis(d)]
    medications = [m for m in medications if _is_real_medication(m)]

    # Add NER-detected symptoms / tests / procedures (with cleaning)
    ner_symptoms = ner_entities.get("symptoms", [])
    ner_tests = [
        t for t in ner_entities.get("tests", [])
        if not _looks_like_noise_ner(t)
    ]
    ner_procedures = [
        p for p in ner_entities.get("procedures", [])
        if not _looks_like_noise_ner(p)
    ]

    logger.info(
        "Extraction: patient=%s, diagnoses=%d, meds=%d, labs=%d, recs=%d | "
        "NER: diseases=%d, meds=%d, symptoms=%d, tests=%d",
        bool(patient_info), len(diagnoses), len(medications),
        len(lab_values), len(recommendations),
        len(ner_entities.get('diseases', [])),
        len(ner_entities.get('medications', [])),
        len(ner_symptoms), len(ner_tests),
    )

    # ── Phase 2: Single T5 pass (only if extraction found gaps) ──
    t5_answers = {}
    has_enough = (patient_info and diagnoses and lab_values)

    if not has_enough:
        # Only run ONE quick T5 call to fill gaps — avoids CPU timeout
        model, tokenizer, device = load_generator_model()
        report_for_t5 = cleaned_text[:1500]
        prompt = (
            "Summarize this medical report briefly. "
            "Include patient name, age, diagnoses, key test results, "
            "medications, and recommendations.\n\n"
            f"Report:\n{report_for_t5}\n\nSummary:"
        )
        try:
            answer = _ask_t5(model, tokenizer, device, prompt, min_tokens=10)
            if answer and len(answer) > 10:
                t5_answers["summary"] = answer
                logger.info("T5 [summary]: %d chars", len(answer))
        except Exception as exc:
            logger.warning("T5 summary failed: %s", exc)
    else:
        logger.info("Skipping T5 — structured extraction sufficient")

    # ── Phase 3: Build the final summary ──
    summary = _build_full_summary(
        patient_info, diagnoses, medications,
        recommendations, lab_values, t5_answers,
        file_name, cleaned_text,
        ner_symptoms=ner_symptoms,
        ner_tests=ner_tests,
        ner_procedures=ner_procedures,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Compose raw T5 output for the frontend
    raw_t5 = " | ".join(f"[{k}] {v}" for k, v in t5_answers.items()) if t5_answers else ""

    return {
        "summary_type": "report_summary",
        "file_name": file_name,
        "summary": summary,
        "raw_t5_summary": raw_t5,
        "report_length": len(cleaned_text),
        "chunks_processed": len(_FOCUSED_QUESTIONS),
        "latency_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Post-process T5 output
# ---------------------------------------------------------------------------

def _clean_t5_output(text: str) -> str:
    if not text:
        return text
    echo_patterns = [
        r"(?i)^you are a (caring |)medical assistant[^.]*\.",
        r"(?i)^summarize the following[^.]*\.",
        r"(?i)^write a clear[^.]*\.",
        r"(?i)^include patient details[^.]*\.",
        r"(?i)^what are the[^:]*:\s*",
        r"(?i)^what is the[^:]*:\s*",
        r"(?i)^list the[^:]*:\s*",
        r"(?i)^medical report:\s*",
        r"(?i)^summary:\s*",
        r"(?i)^report:\s*",
    ]
    cleaned = text
    for pat in echo_patterns:
        cleaned = re.sub(pat, "", cleaned).strip()
    if len(cleaned) < 5:
        return ""
    return cleaned


# ---------------------------------------------------------------------------
# Build final human-readable summary
# ---------------------------------------------------------------------------

def _build_full_summary(
    patient_info: dict,
    diagnoses: list[str],
    medications: list[str],
    recommendations: list[str],
    lab_values: list[dict],
    t5_answers: dict,
    file_name: str,
    original_text: str,
    ner_symptoms: list[str] = None,
    ner_tests: list[str] = None,
    ner_procedures: list[str] = None,
) -> str:
    """Assemble a human-centric, conversational medical report summary."""
    parts = []

    # ── Conversational opening with patient context ──
    name = patient_info.get("name", "")
    age = patient_info.get("age", "")
    gender = patient_info.get("gender", "")
    date = patient_info.get("date", "")
    pid = patient_info.get("patient_id", "")

    if name and age:
        who = f"{name}, a {age}-year-old {gender.lower()}" if gender else f"{name}, age {age}"
        opener = f"Here's a summary of the medical report for **{who}**"
    elif name:
        opener = f"Here's a summary of the medical report for **{name}**"
    elif age:
        who = f"a {age}-year-old {gender.lower()} patient" if gender else f"a {age}-year-old patient"
        opener = f"Here's a summary of the medical report for {who}"
    else:
        opener = "Here's a summary of your medical report"

    if file_name:
        opener += f" ({file_name})"
    if date:
        opener += f", dated {date}"
    opener += ".\n"
    parts.append(opener)

    # ── What was found (diagnoses) ──
    if diagnoses:
        if len(diagnoses) == 1:
            parts.append(f"**Diagnosis:** {diagnoses[0]}")
        else:
            parts.append("**Diagnoses & Conditions:**")
            for d in diagnoses[:8]:
                parts.append(f"  - {d}")
        parts.append("")

    # ── Symptoms ──
    if ner_symptoms:
        symptom_str = ", ".join(s.lower() for s in ner_symptoms[:6])
        parts.append(f"**Reported Symptoms:** {symptom_str}.")
        parts.append("")

    # ── Vitals & Lab Results (separated for readability) ──
    if lab_values:
        # Separate vitals from lab tests
        vital_labels = {"Blood Pressure", "Heart Rate", "Respiratory Rate",
                        "Temperature", "SpO2", "Weight", "Height", "BMI"}
        vitals = [lv for lv in lab_values if lv["label"] in vital_labels]
        labs = [lv for lv in lab_values if lv["label"] not in vital_labels]

        if vitals:
            parts.append("**Vitals:**")
            for lv in vitals:
                line = f"  - {lv['label']}: {lv['value']} {lv['unit']}"
                if lv.get("note"):
                    line += f" ({lv['note']})"
                parts.append(line)
            parts.append("")

        if labs:
            parts.append("**Lab Results:**")
            for lv in labs:
                line = f"  - {lv['label']}: {lv['value']} {lv['unit']}"
                if lv.get("note"):
                    line += f" — {lv['note']}"
                parts.append(line)
            parts.append("")

    # ── Tests & Investigations (NER-detected, excluding duplicates) ──
    if ner_tests:
        lab_labels = {lv["label"].lower() for lv in lab_values} if lab_values else set()
        extra_tests = [t for t in ner_tests if t.lower() not in lab_labels]
        if extra_tests:
            test_str = ", ".join(extra_tests[:5])
            parts.append(f"**Investigations Mentioned:** {test_str}.")
            parts.append("")

    # ── Medications ──
    if medications:
        parts.append("**Medications:**")
        for med in medications[:8]:
            parts.append(f"  - {med}")
        parts.append("")

    # ── Procedures ──
    if ner_procedures:
        proc_str = ", ".join(ner_procedures[:4])
        parts.append(f"**Procedures:** {proc_str}.")
        parts.append("")

    # ── Recommendations (as a readable paragraph when short) ──
    if recommendations:
        if len(recommendations) == 1:
            parts.append(f"**Recommendation:** {recommendations[0]}.")
        else:
            parts.append("**Recommendations:**")
            for rec in recommendations[:5]:
                parts.append(f"  - {rec}")
        parts.append("")

    # ── T5 supplemental info ──
    if t5_answers.get("summary"):
        parts.append(f"**Additional Notes:** {t5_answers['summary']}")
        parts.append("")

    # ── Fallback if nothing was extracted ──
    if not any([patient_info, diagnoses, medications, lab_values,
                recommendations, t5_answers, ner_symptoms]):
        parts.append("I couldn't fully parse this report's structure. Here's the raw content:\n")
        lines = [l.strip() for l in original_text.split('\n') if l.strip()]
        for line in lines[:10]:
            parts.append(f"  {line}")
        parts.append("")
        parts.append("For better results, try uploading a text-based PDF or a clearer image.")
        parts.append("")

    # ── Closing ──
    parts.append(
        "---\n"
        "*This is an AI-generated summary for informational purposes only. "
        "Please consult your healthcare provider for medical advice.*"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Value interpretation helpers
# ---------------------------------------------------------------------------

def _get_value_note(label: str, value_str: str) -> str:
    """Return a brief contextual note for common lab values."""
    try:
        val = float(value_str.replace(",", "").split()[0])
    except (ValueError, IndexError):
        return ""

    notes = {
        "Hemoglobin": lambda v: (
            "Below normal (normal: 12–17 g/dL)" if v < 12
            else "Above normal" if v > 17
            else "Normal"
        ),
        "Blood Sugar": lambda v: (
            "Low" if v < 70 else "Normal" if v <= 140
            else "Elevated – may indicate diabetes" if v <= 200
            else "High – consult a doctor"
        ),
        "Cholesterol": lambda v: (
            "Desirable (<200)" if v < 200
            else "Borderline high (200-239)" if v <= 239
            else "High – needs attention"
        ),
        "Creatinine": lambda v: (
            "Normal (0.6–1.2)" if 0.6 <= v <= 1.2
            else "Abnormal – kidney evaluation needed"
        ),
        "Bilirubin": lambda v: (
            "Normal (≤1.2)" if v <= 1.2
            else "Elevated – possible liver issue"
        ),
        "TSH": lambda v: (
            "Low – possible hyperthyroidism" if v < 0.4
            else "Normal (0.4–4.0)" if v <= 4.0
            else "Elevated – possible hypothyroidism"
        ),
        "HbA1c": lambda v: (
            "Normal (<5.7%)" if v < 5.7
            else "Pre-diabetic (5.7–6.4%)" if v <= 6.4
            else "Diabetic range (>6.4%)"
        ),
        "SGOT/AST": lambda v: (
            "Normal (≤40)" if v <= 40
            else "Elevated – liver function concern"
        ),
        "SGPT/ALT": lambda v: (
            "Normal (≤40)" if v <= 40
            else "Elevated – liver function concern"
        ),
        "Blood Urea": lambda v: (
            "Normal (7–20)" if 7 <= v <= 20
            else "Abnormal"
        ),
        "Uric Acid": lambda v: (
            "Normal (3–7)" if 3 <= v <= 7
            else "Abnormal"
        ),
        "Sodium": lambda v: (
            "Normal (136–145)" if 136 <= v <= 145
            else "Abnormal"
        ),
        "Potassium": lambda v: (
            "Normal (3.5–5.0)" if 3.5 <= v <= 5.0
            else "Abnormal – needs attention"
        ),
        "SpO2": lambda v: (
            "Normal (≥95%)" if v >= 95
            else "Low – monitor closely" if v >= 90
            else "Critically low – seek help"
        ),
        "BMI": lambda v: (
            "Underweight" if v < 18.5
            else "Normal" if v < 25
            else "Overweight" if v < 30
            else "Obese"
        ),
        "LDL Cholesterol": lambda v: (
            "Optimal (<100)" if v < 100
            else "Near optimal (100-129)" if v < 130
            else "Borderline high (130-159)" if v < 160
            else "High"
        ),
        "HDL Cholesterol": lambda v: (
            "Low – risk factor" if v < 40
            else "Normal" if v < 60
            else "Optimal (protective)"
        ),
        "Triglycerides": lambda v: (
            "Normal (<150)" if v < 150
            else "Borderline high (150-199)" if v < 200
            else "High"
        ),
        "Blood Pressure": lambda _v: "",  # Handled separately
        "Heart Rate": lambda v: (
            "Low (bradycardia)" if v < 60
            else "Normal" if v <= 100
            else "Elevated (tachycardia)"
        ),
    }

    # Special handling for blood pressure
    if label == "Blood Pressure":
        try:
            sys_str, dia_str = value_str.split("/")
            sys_val = int(sys_str.strip())
            dia_val = int(dia_str.strip())
            if sys_val < 120 and dia_val < 80:
                return "Normal"
            elif sys_val < 130 and dia_val < 80:
                return "Elevated"
            elif sys_val < 140 or dia_val < 90:
                return "High (Stage 1 Hypertension)"
            else:
                return "High (Stage 2 Hypertension)"
        except (ValueError, AttributeError):
            return ""

    fn = notes.get(label)
    return fn(val) if fn else ""
