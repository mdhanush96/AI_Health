"""
MedAI – Medical Report Summarizer

Pipeline:
  User Upload → Text Extraction (PDF/OCR) → Section Detection →
  Medical Entity Extraction → Structured JSON → LLM Summary → Frontend

This module handles stages 2-6.  Stage 1 (file upload) lives in views.py,
and the raw text arrives as ``report_text``.
"""

import logging
import re
import time

import torch

from ml_engine import config
from ml_engine.rag.rag_loader import load_generator_model

logger = logging.getLogger("ml_engine")

# ═══════════════════════════════════════════════════════════════════════════
# 1. SUMMARIZATION INTENT DETECTION
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# 2. TEXT PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def _clean_report_text(raw_text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', raw_text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# 3. SECTION DETECTION  — regex-based section parser
# ═══════════════════════════════════════════════════════════════════════════

# Ordered list of (section_key, header‑regex).
# The parser walks the list and captures text between consecutive headers.
_SECTION_HEADERS = [
    ("demographics",   r"(?:Patient\s*(?:Demographics|Information|Details)|Personal\s*(?:Details|Information))"),
    ("lifestyle",      r"(?:Patient\s*)?Lifestyle(?:\s*(?:Data|Information|Details))?"),
    ("vitals",         r"(?:Patient\s*)?Vitals?(?:\s*(?:Signs?|Data|Information))?"),
    ("doctor_info",    r"(?:Doctor(?:'?s?)?\s*(?:Information|Details)|Attending\s*(?:Physician|Doctor))"),
    ("doctor_notes",   r"(?:Doctor(?:'?s?)?\s*(?:Notes?|Assessment|Comments?|Remarks?)|Clinical\s*(?:Notes?|Assessment|Impression))"),
    ("conditions",     r"(?:Medical\s*Conditions?|Diagnos[ei]s|Conditions?|Primary\s*Diagnosis)"),
    ("medications",    r"(?:Current\s*)?Medications?(?:\s*(?:List|Prescribed))?|Prescription|Drugs?|Medicines?|Rx"),
    ("tests",          r"(?:Medical\s*)?(?:Tests?|Investigations?|Lab(?:oratory)?\s*(?:Results?|Tests?|Findings?))"),
    ("past_visits",    r"(?:Past\s*)?Hospital\s*Visits?|(?:Past\s*)?Medical\s*History|Visit\s*History"),
    ("recommendations",r"(?:Recommend(?:ations?|ed)|Follow[\s-]*up|Advice|Instructions?|Plan)"),
]


def _detect_sections(text: str) -> dict[str, str]:
    """
    Split the report text into named sections based on header patterns.
    Returns a dict  {section_key: section_body_text}.
    Any text that doesn't fall under a recognized header goes to ``"other"``.
    """
    # Build a combined pattern that matches any known header
    header_pats = [(key, re.compile(pat, re.IGNORECASE)) for key, pat in _SECTION_HEADERS]

    # Find all header positions
    markers: list[tuple[int, int, str]] = []  # (start, end, key)
    for key, pat in header_pats:
        for m in pat.finditer(text):
            markers.append((m.start(), m.end(), key))
    markers.sort(key=lambda x: x[0])

    sections: dict[str, str] = {}

    if not markers:
        # No recognizable sections — put everything under "other"
        sections["other"] = text
        return sections

    # Text before first header
    preamble = text[: markers[0][0]].strip()
    if preamble:
        sections["preamble"] = preamble

    for i, (start, end, key) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        body = text[end:next_start].strip().lstrip(":").strip()
        if body:
            # If same key appears twice, append
            if key in sections:
                sections[key] += "\n" + body
            else:
                sections[key] = body

    return sections


# ═══════════════════════════════════════════════════════════════════════════
# 4. MEDICAL ENTITY EXTRACTION  — per-section structured extraction
# ═══════════════════════════════════════════════════════════════════════════

# ── 4a. Demographics ──

def _extract_demographics(text: str) -> dict:
    info: dict = {}

    for pat in [
        r"(?:(?:Full\s*)?Name|Patient(?:\s*Name)?)\s*[:\-]\s*(.+)",
        r"(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            name = m.group(1).strip().rstrip(".")
            # Reject obvious non-names
            if len(name) > 2 and not re.match(r"(?i)(summary|report|test|result)", name):
                info["name"] = name
                break

    m = re.search(r'(?:age|aged)\s*[:\-]?\s*(\d{1,3})', text, re.I)
    if not m:
        m = re.search(r'(\d{1,3})\s*(?:years?|yrs?)\b', text, re.I)
    if m:
        info["age"] = m.group(1)

    m = re.search(r'(?:gender|sex)\s*[:\-]?\s*(male|female|m|f)\b', text, re.I)
    if m:
        g = m.group(1).strip().lower()
        info["gender"] = "Male" if g in ("m", "male") else "Female"
    else:
        if re.search(r'\bmale\b', text, re.I):
            info["gender"] = "Male"
        elif re.search(r'\bfemale\b', text, re.I):
            info["gender"] = "Female"

    for label, pats in [
        ("dob", [
            r'(?:D\.?O\.?B\.?|Date\s*of\s*Birth|DOB)\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:D\.?O\.?B\.?|Date\s*of\s*Birth|DOB)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})',
        ]),
        ("date", [
            r'(?:date|dated|report\s*date|visit\s*date)\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:date|dated)\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})',
        ]),
        ("patient_id", [
            r'(?:patient\s*id|mrn|medical\s*record|record\s*no|uhid|id)\s*[:\-#]?\s*(\w[\w\-]+)',
        ]),
        ("blood_group", [
            r'(?:blood\s*(?:group|type))\s*[:\-]?\s*(A|B|AB|O)\s*([+-]?(?:ve)?)',
        ]),
        ("phone", [
            r'(?:phone|contact|mobile|tel)\s*[:\-]?\s*([\d\s\-+()]{7,15})',
        ]),
        ("address", [
            r'(?:address)\s*[:\-]?\s*(.+)',
        ]),
    ]:
        for pat in pats:
            m = re.search(pat, text, re.I)
            if m:
                val = " ".join(g for g in m.groups() if g).strip()
                if val:
                    info[label] = val
                break

    return info


# ── 4b. Lifestyle ──

def _extract_lifestyle(text: str) -> dict:
    lifestyle: dict = {}
    for field, pats in [
        ("smoking", [r'(?:smoking|tobacco)\s*(?:status|habit)?[:\-]?\s*([^\n,]+)']),
        ("alcohol", [r'(?:alcohol|drinking)\s*(?:status|consumption|habit)?[:\-]?\s*([^\n,]+)']),
        ("diet", [r'(?:diet|dietary)\s*(?:preference|type|habit)?[:\-]?\s*([^\n,]+)']),
        ("exercise", [r'(?:exercise|physical\s*activity)\s*(?:level|habit)?[:\-]?\s*([^\n,]+)']),
    ]:
        for pat in pats:
            m = re.search(pat, text, re.I)
            if m:
                lifestyle[field] = m.group(1).strip().rstrip(".")
                break
    return lifestyle


# ── 4c. Vitals ──

_VITAL_PATTERNS = [
    ("blood_pressure", r'(?:blood\s*pressure|bp)\s*[:\-]?\s*(\d{2,3}\s*/\s*\d{2,3})\s*(mmhg)?', "mmHg"),
    ("heart_rate",     r'(?:heart\s*rate|pulse|hr)\s*[:\-]?\s*(\d{2,3})\s*(?:bpm|/min)?', "bpm"),
    ("respiratory_rate", r'(?:respiratory\s*rate|rr|resp\.?\s*rate)\s*[:\-]?\s*(\d{1,2})\s*(?:/min|breaths)?', "/min"),
    ("temperature",    r'(?:temperature|temp)\s*[:\-]?\s*([\d.]+)\s*°?\s*([fFcC])?', "°C"),
    ("spo2",           r'(?:spo2|oxygen\s*saturation|o2\s*sat)\s*[:\-]?\s*(\d{2,3})\s*(%)?', "%"),
    ("weight",         r'(?:weight|wt)\s*[:\-]?\s*([\d.]+)\s*(kg|lbs?)?', "kg"),
    ("height",         r'(?:height|ht)\s*[:\-]?\s*([\d.]+)\s*(cm|m|ft|feet)?', "cm"),
    ("bmi",            r'(?:bmi|body\s*mass\s*index)\s*[:\-]?\s*([\d.]+)', "kg/m²"),
]


def _extract_vitals(text: str) -> list[dict]:
    vitals = []
    text_lower = text.lower()
    for name, pat, default_unit in _VITAL_PATTERNS:
        m = re.search(pat, text_lower)
        if m:
            value = m.group(1).strip()
            unit = (m.group(2).strip() if m.lastindex >= 2 and m.group(2) else default_unit)
            vitals.append({"name": name, "value": value, "unit": unit})
    return vitals


# ── 4d. Conditions / Diagnoses ──

_COMMON_CONDITIONS = [
    r'type\s*[12]\s*diabetes\s*(?:mellitus)?',
    r'hypertension', r'hyperthyroid(?:ism)?', r'hypothyroid(?:ism)?',
    r'anemia|anaemia',  r'pneumonia', r'bronchitis', r'asthma',
    r'coronary\s*artery\s*disease', r'heart\s*failure',
    r'chronic\s*kidney\s*disease', r'renal\s*failure',
    r'liver\s*(?:disease|cirrhosis|failure)', r'hepatitis\s*[a-e]?',
    r'dengue', r'malaria', r'typhoid',  r'tuberculosis|tb\b',
    r'urinary\s*tract\s*infection|uti\b',
    r'dyslipidemia|hyperlipidemia', r'obesity', r'osteoporosis',
    r'arthritis', r'gout', r'migraine', r'epilepsy',
    r'depression', r'anxiety\s*disorder', r'copd',
    r'peripheral\s*neuropathy',
    r'diabetic\s*(?:neuropathy|retinopathy|nephropathy)',
]


def _extract_conditions(text: str) -> list[str]:
    conditions: list[str] = []

    # Bullet / numbered list items (most reliable if present)
    items = re.findall(r'[•\-\d.]+[.)]*\s*(.+)', text)
    for item in items:
        item = item.strip().rstrip(".")
        if 4 < len(item) < 120:
            conditions.append(item)

    # Keyword scan (fallback)
    seen = {c.lower() for c in conditions}
    for pat in _COMMON_CONDITIONS:
        m = re.search(pat, text, re.I)
        if m and m.group(0).strip().lower() not in seen:
            conditions.append(m.group(0).strip().title())
            seen.add(m.group(0).strip().lower())

    return conditions


# ── 4e. Medications ──

_KNOWN_DRUG_SUFFIXES = (
    "min", "pin", "pril", "olol", "artan", "statin", "zole",
    "mycin", "cillin", "parin", "mab", "nib", "ide", "ine",
    "pam", "lam", "oxin", "done", "vir", "fen",
)


def _extract_medications(text: str) -> list[dict]:
    medications: list[dict] = []

    # Pattern: "Drug Name – dose – frequency"  or  "Drug Name  dose  frequency"
    med_line_pat = re.compile(
        r'[•\-\d.]*[.)]*\s*'                        # optional bullet
        r'([A-Za-z][A-Za-z0-9\s]{2,30}?)'           # drug name
        r'\s*[\-–—|:]*\s*'                           # separator
        r'(\d+[\d,.]*\s*(?:mg|mcg|g|ml|IU|units?)?)'  # dose
        r'(?:\s*[\-–—|:]*\s*'                        # separator
        r'([^\n]{3,40}))?',                          # frequency (optional)
        re.I,
    )

    for m in med_line_pat.finditer(text):
        name = m.group(1).strip().rstrip("-–— ")
        dose = m.group(2).strip()
        freq = (m.group(3).strip().rstrip(".") if m.group(3) else "")
        if len(name) >= 3:
            medications.append({"name": name, "dose": dose, "frequency": freq})

    # If structured parsing found nothing, try line-by-line
    if not medications:
        for line in text.split("\n"):
            line = line.strip().lstrip("•-0123456789.) ")
            if not line:
                continue
            parts = re.split(r'\s*[\-–—|]+\s*', line, maxsplit=2)
            if parts and len(parts[0]) >= 3:
                entry: dict = {"name": parts[0].strip()}
                if len(parts) > 1:
                    entry["dose"] = parts[1].strip()
                if len(parts) > 2:
                    entry["frequency"] = parts[2].strip().rstrip(".")
                medications.append(entry)

    return medications


# ── 4f. Lab Tests ──

_LAB_PATTERNS = [
    (r'(?:hemoglobin|hb)\s*[:\-]?\s*([\d.]+)\s*(g/dl|gm/dl|g%)?', 'Hemoglobin', 'g/dL'),
    (r'(?:wbc|white\s*blood\s*cell)\s*(?:count)?\s*[:\-]?\s*([\d,.]+)', 'WBC Count', 'cells/cumm'),
    (r'(?:rbc|red\s*blood\s*cell)\s*(?:count)?\s*[:\-]?\s*([\d,.]+)', 'RBC Count', 'million/cumm'),
    (r'(?:platelet)\s*(?:count)?\s*[:\-]?\s*([\d,.]+)', 'Platelet Count', '/cumm'),
    (r'(?:blood\s*sugar|glucose|fasting\s*(?:blood\s*)?sugar|fbs|rbs)\s*[:\-]?\s*([\d.]+)', 'Blood Sugar', 'mg/dL'),
    (r'(?:creatinine)\s*[:\-]?\s*([\d.]+)', 'Creatinine', 'mg/dL'),
    (r'(?:total\s*)?cholesterol\s*[:\-]?\s*([\d.]+)', 'Cholesterol', 'mg/dL'),
    (r'(?:total\s*)?bilirubin\s*[:\-]?\s*([\d.]+)', 'Bilirubin', 'mg/dL'),
    (r'(?:tsh|thyroid)\s*[:\-]?\s*([\d.]+)', 'TSH', 'mIU/L'),
    (r'(?:hba1c|a1c|glycated\s*h(?:ae)?moglobin)\s*[:\-]?\s*([\d.]+)', 'HbA1c', '%'),
    (r'(?:sgot|ast)\s*[:\-]?\s*([\d.]+)', 'SGOT/AST', 'U/L'),
    (r'(?:sgpt|alt)\s*[:\-]?\s*([\d.]+)', 'SGPT/ALT', 'U/L'),
    (r'(?:urea|blood\s*urea)\s*[:\-]?\s*([\d.]+)', 'Blood Urea', 'mg/dL'),
    (r'(?:uric\s*acid)\s*[:\-]?\s*([\d.]+)', 'Uric Acid', 'mg/dL'),
    (r'(?:sodium|na)\s*[:\-]?\s*([\d.]+)', 'Sodium', 'mEq/L'),
    (r'(?:potassium|k)\s*[:\-]?\s*([\d.]+)', 'Potassium', 'mEq/L'),
    (r'(?:calcium|ca)\s*[:\-]?\s*([\d.]+)', 'Calcium', 'mg/dL'),
    (r'(?:ldl)\s*(?:cholesterol)?\s*[:\-]?\s*([\d.]+)', 'LDL', 'mg/dL'),
    (r'(?:hdl)\s*(?:cholesterol)?\s*[:\-]?\s*([\d.]+)', 'HDL', 'mg/dL'),
    (r'(?:triglyceride)s?\s*[:\-]?\s*([\d.]+)', 'Triglycerides', 'mg/dL'),
    (r'(?:esr|sed\s*rate)\s*[:\-]?\s*([\d.]+)', 'ESR', 'mm/hr'),
    (r'(?:albumin)\s*[:\-]?\s*([\d.]+)', 'Albumin', 'g/dL'),
    (r'(?:vitamin\s*d|vit\s*d)\s*[:\-]?\s*([\d.]+)', 'Vitamin D', 'ng/mL'),
    (r'(?:ferritin)\s*[:\-]?\s*([\d.]+)', 'Ferritin', 'ng/mL'),
    (r'(?:iron|serum\s*iron)\s*[:\-]?\s*([\d.]+)', 'Iron', 'mcg/dL'),
]


def _extract_lab_tests(text: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    text_lower = text.lower()
    for pat, label, default_unit in _LAB_PATTERNS:
        m = re.search(pat, text_lower)
        if m and label not in seen:
            value = m.group(1).strip()
            if not any(c.isdigit() for c in value):
                continue
            unit = (m.group(2).strip() if m.lastindex >= 2 and m.group(2) else default_unit)
            results.append({"test": label, "value": value, "unit": unit})
            seen.add(label)

    # Also grab tabular lines:  "TestName   value   unit   refRange"
    table_pat = re.compile(
        r'([A-Za-z][A-Za-z /()]+?)\s+'
        r'(\d+\.?\d*)\s+'
        r'([a-zA-Z/%µμ]+(?:/[a-zA-Z]+)?)\s+'
        r'([\d.\-–]+)',
    )
    for m in table_pat.finditer(text):
        label = m.group(1).strip().title()
        if label not in seen and len(label) > 2:
            results.append({
                "test": label,
                "value": m.group(2).strip(),
                "unit": m.group(3).strip(),
                "reference": m.group(4).strip(),
            })
            seen.add(label)

    return results


# ── 4g. Doctor notes / assessment ──

def _extract_doctor_notes(text: str) -> str:
    # Collapse bullet-list into paragraph
    lines = [l.strip().lstrip("•-") .strip() for l in text.split("\n") if l.strip()]
    return " ".join(lines)


# ── 4h. Past visits ──

def _extract_past_visits(text: str) -> list[dict]:
    visits: list[dict] = []
    # Look for date + reason pairs
    visit_pat = re.compile(
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\s*[\-–—|:]*\s*(.+)',
    )
    for m in visit_pat.finditer(text):
        visits.append({"date": m.group(1).strip(), "reason": m.group(2).strip().rstrip(".")})

    # Fallback: bullet items
    if not visits:
        items = re.findall(r'[•\-\d.]+[.)]*\s*(.+)', text)
        for item in items:
            item = item.strip()
            if len(item) > 5:
                visits.append({"reason": item})

    return visits


# ── 4i. Recommendations / follow-up ──

def _extract_recommendations(text: str) -> list[str]:
    recs: list[str] = []
    items = re.findall(r'[•\-\d.]+[.)]*\s*(.+)', text)
    for item in items:
        item = item.strip().rstrip(".")
        if len(item) > 5:
            recs.append(item)

    # Sentence-level fallback
    if not recs:
        sentences = re.split(r'[.\n]', text)
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                recs.append(s)

    return recs


# ═══════════════════════════════════════════════════════════════════════════
# 5. BUILD STRUCTURED JSON  — assemble all extracted data
# ═══════════════════════════════════════════════════════════════════════════

def _build_medical_json(text: str) -> dict:
    """
    Parse the full report text into a structured dict.
    Works with sectioned reports AND unsectioned ones (falls back to
    running all extractors on the full text).
    """
    sections = _detect_sections(text)
    data: dict = {}

    # Demographics
    demo_text = sections.get("demographics") or sections.get("preamble", "")
    if not demo_text:
        demo_text = text[:600]  # first part often has patient info
    data["patient"] = _extract_demographics(demo_text)
    # If demographics missed, try full text
    if not data["patient"]:
        data["patient"] = _extract_demographics(text)

    # Conditions
    cond_text = sections.get("conditions", "")
    # Also check doctor notes for diagnoses
    doc_notes_text = sections.get("doctor_notes", "")
    data["conditions"] = _extract_conditions(cond_text + "\n" + doc_notes_text) if (cond_text or doc_notes_text) else _extract_conditions(text)

    # Vitals
    vitals_text = sections.get("vitals", "")
    data["vitals"] = _extract_vitals(vitals_text) if vitals_text else _extract_vitals(text)

    # Lifestyle
    lifestyle_text = sections.get("lifestyle", "")
    data["lifestyle"] = _extract_lifestyle(lifestyle_text) if lifestyle_text else _extract_lifestyle(text)

    # Doctor notes / assessment
    if doc_notes_text:
        data["doctor_notes"] = _extract_doctor_notes(doc_notes_text)

    # Medications
    med_text = sections.get("medications", "")
    data["medications"] = _extract_medications(med_text) if med_text else _extract_medications(text)

    # Lab tests
    test_text = sections.get("tests", "")
    data["lab_tests"] = _extract_lab_tests(test_text) if test_text else _extract_lab_tests(text)

    # Past visits
    visit_text = sections.get("past_visits", "")
    if visit_text:
        data["past_visits"] = _extract_past_visits(visit_text)

    # Recommendations
    rec_text = sections.get("recommendations", "")
    if rec_text:
        data["recommendations"] = _extract_recommendations(rec_text)

    # Doctor info
    doc_info_text = sections.get("doctor_info", "")
    if doc_info_text:
        data["doctor_info"] = doc_info_text.strip()

    return data


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM SUMMARY GENERATION  — prompt-guided FLAN-T5 on structured data
# ═══════════════════════════════════════════════════════════════════════════

def _structured_to_prompt_text(data: dict) -> str:
    """Convert the structured JSON into a clean prompt-friendly text block."""
    parts: list[str] = []

    p = data.get("patient", {})
    if p:
        demo_parts = []
        if p.get("name"):
            demo_parts.append(f"Name: {p['name']}")
        if p.get("age"):
            demo_parts.append(f"Age: {p['age']}")
        if p.get("gender"):
            demo_parts.append(f"Gender: {p['gender']}")
        if p.get("dob"):
            demo_parts.append(f"DOB: {p['dob']}")
        if demo_parts:
            parts.append("Patient: " + ", ".join(demo_parts))

    if data.get("conditions"):
        parts.append("Conditions: " + "; ".join(data["conditions"]))

    if data.get("vitals"):
        v_strs = [f"{v['name'].replace('_',' ').title()}: {v['value']} {v['unit']}" for v in data["vitals"]]
        parts.append("Vitals: " + ", ".join(v_strs))

    if data.get("lifestyle"):
        l_strs = [f"{k.title()}: {v}" for k, v in data["lifestyle"].items()]
        parts.append("Lifestyle: " + ", ".join(l_strs))

    if data.get("doctor_notes"):
        parts.append(f"Doctor Assessment: {data['doctor_notes'][:300]}")

    if data.get("medications"):
        m_strs = []
        for med in data["medications"][:10]:
            s = med["name"]
            if med.get("dose"):
                s += f" {med['dose']}"
            if med.get("frequency"):
                s += f" {med['frequency']}"
            m_strs.append(s)
        parts.append("Medications: " + "; ".join(m_strs))

    if data.get("lab_tests"):
        t_strs = [f"{t['test']}: {t['value']} {t['unit']}" for t in data["lab_tests"][:12]]
        parts.append("Lab Results: " + ", ".join(t_strs))

    if data.get("recommendations"):
        parts.append("Recommendations: " + "; ".join(data["recommendations"][:6]))

    return "\n".join(parts)


_LLM_SUMMARY_PROMPT = (
    "Summarize the following medical report in a clear, structured format.\n"
    "Include:\n"
    "- Patient details (name, age, gender)\n"
    "- Diagnosed conditions\n"
    "- Key vitals\n"
    "- Medications with dosage\n"
    "- Doctor's assessment\n"
    "- Recommended follow-up\n\n"
    "Medical Data:\n{structured_text}\n\n"
    "Structured Summary:"
)


def _run_llm_summary(structured_text: str) -> str:
    """Run prompt-guided FLAN-T5 on the pre-structured medical data."""
    model, tokenizer, device = load_generator_model()

    prompt = _LLM_SUMMARY_PROMPT.format(structured_text=structured_text[:2000])

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
            max_new_tokens=config.GENERATOR_MAX_OUTPUT_TOKENS,
            min_new_tokens=20,
            no_repeat_ngram_size=4,
            early_stopping=True,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
        )

    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return _clean_t5_output(raw)


def _clean_t5_output(text: str) -> str:
    if not text:
        return text
    echo_patterns = [
        r"(?i)^you are a (?:caring )?medical assistant[^.]*\.",
        r"(?i)^summarize the following[^.]*\.",
        r"(?i)^write a clear[^.]*\.",
        r"(?i)^include[^.]*\.",
        r"(?i)^medical data:\s*",
        r"(?i)^summary:\s*",
        r"(?i)^structured summary:\s*",
        r"(?i)^report:\s*",
    ]
    cleaned = text
    for pat in echo_patterns:
        cleaned = re.sub(pat, "", cleaned).strip()
    return cleaned if len(cleaned) >= 5 else ""


# ═══════════════════════════════════════════════════════════════════════════
# 7. RENDER FINAL STRUCTURED OUTPUT  — doctor-quality formatted summary
# ═══════════════════════════════════════════════════════════════════════════

def _format_vital_label(name: str) -> str:
    labels = {
        "blood_pressure": "Blood Pressure",
        "heart_rate": "Heart Rate",
        "respiratory_rate": "Respiratory Rate",
        "temperature": "Temperature",
        "spo2": "Oxygen Saturation",
        "weight": "Weight",
        "height": "Height",
        "bmi": "BMI",
    }
    return labels.get(name, name.replace("_", " ").title())


def _render_structured_output(data: dict, llm_summary: str, file_name: str | None) -> str:
    """
    Render the final summary in a clean, doctor-friendly format.
    Uses the structured JSON data as the primary source,
    with the LLM summary as supplementary context.
    """
    parts: list[str] = []

    # ── Title ──
    parts.append("**PATIENT SUMMARY**\n")

    # ── Patient Information ──
    p = data.get("patient", {})
    if p:
        parts.append("**Patient Information**")
        if p.get("name"):
            parts.append(f"  Name: {p['name']}")
        if p.get("age"):
            parts.append(f"  Age: {p['age']}")
        if p.get("gender"):
            parts.append(f"  Gender: {p['gender']}")
        if p.get("dob"):
            parts.append(f"  DOB: {p['dob']}")
        if p.get("patient_id"):
            parts.append(f"  Patient ID: {p['patient_id']}")
        if p.get("blood_group"):
            parts.append(f"  Blood Group: {p['blood_group']}")
        if p.get("phone"):
            parts.append(f"  Contact: {p['phone']}")
        parts.append("")

    # ── Medical Conditions ──
    if data.get("conditions"):
        parts.append("**Medical Conditions**")
        for c in data["conditions"]:
            parts.append(f"  • {c}")
        parts.append("")

    # ── Vitals ──
    if data.get("vitals"):
        date_str = p.get("date", "")
        header = f"**Vitals ({date_str})**" if date_str else "**Vitals**"
        parts.append(header)
        for v in data["vitals"]:
            parts.append(f"  • {_format_vital_label(v['name'])}: {v['value']} {v['unit']}")
        parts.append("")

    # ── Lifestyle ──
    if data.get("lifestyle"):
        parts.append("**Lifestyle**")
        for k, v in data["lifestyle"].items():
            parts.append(f"  • {k.title()}: {v}")
        parts.append("")

    # ── Doctor Assessment ──
    if data.get("doctor_notes"):
        parts.append("**Doctor Assessment**")
        parts.append(f"  {data['doctor_notes']}")
        parts.append("")

    # ── Medications ──
    if data.get("medications"):
        parts.append("**Medications**")
        for med in data["medications"]:
            line = f"  • {med['name']}"
            if med.get("dose"):
                line += f" – {med['dose']}"
            if med.get("frequency"):
                line += f" – {med['frequency']}"
            parts.append(line)
        parts.append("")

    # ── Lab Results ──
    if data.get("lab_tests"):
        parts.append("**Lab Results**")
        for t in data["lab_tests"]:
            line = f"  • {t['test']}: {t['value']} {t['unit']}"
            if t.get("reference"):
                line += f" (ref: {t['reference']})"
            parts.append(line)
        parts.append("")

    # ── Past Visits ──
    if data.get("past_visits"):
        parts.append("**Past Hospital Visits**")
        for v in data["past_visits"]:
            if v.get("date"):
                parts.append(f"  • {v['date']} — {v.get('reason', '')}")
            else:
                parts.append(f"  • {v.get('reason', '')}")
        parts.append("")

    # ── Recommendations ──
    if data.get("recommendations"):
        parts.append("**Recommended Follow-up**")
        for rec in data["recommendations"]:
            parts.append(f"  • {rec}")
        parts.append("")

    # ── LLM Additional Notes (only if it adds value beyond extracted data) ──
    has_good_extraction = (
        data.get("patient") and (data.get("conditions") or data.get("lab_tests"))
    )
    if llm_summary and len(llm_summary) > 20 and not has_good_extraction:
        parts.append("**Additional Notes**")
        parts.append(f"  {llm_summary}")
        parts.append("")

    # ── Fallback if nothing was extracted ──
    if not any([p, data.get("conditions"), data.get("medications"),
                data.get("vitals"), data.get("lab_tests")]):
        if llm_summary:
            parts.append(llm_summary)
            parts.append("")
        else:
            parts.append("Could not parse this report's structure. Please try a clearer file.")
            parts.append("")

    # ── Disclaimer ──
    parts.append(
        "---\n"
        "*This is an AI-generated summary for informational purposes only. "
        "Please consult your healthcare provider for medical advice.*"
    )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def summarize_report(report_text: str, file_name: str = None) -> dict:
    """
    Summarize a medical report.

    Pipeline:
      Text Extraction (done) → Section Detection → Medical Entity Extraction
      → Structured JSON → LLM Summary → Structured Output
    """
    t0 = time.perf_counter()

    # ── Stage 1: Clean text ──
    cleaned_text = _clean_report_text(report_text)

    if len(cleaned_text) < 20:
        return {
            "summary_type": "report_summary",
            "file_name": file_name,
            "summary": (
                "The uploaded file doesn't seem to contain enough text to summarise. "
                "Please upload a clearer report or describe what you'd like to know."
            ),
            "structured_data": {},
            "report_length": len(cleaned_text),
            "latency_ms": 0,
        }

    # ── Stage 2-4: Section Detection → Entity Extraction → Structured JSON ──
    medical_data = _build_medical_json(cleaned_text)

    logger.info(
        "Extraction: patient=%s, conditions=%d, vitals=%d, meds=%d, labs=%d",
        bool(medical_data.get("patient")),
        len(medical_data.get("conditions", [])),
        len(medical_data.get("vitals", [])),
        len(medical_data.get("medications", [])),
        len(medical_data.get("lab_tests", [])),
    )

    # ── Stage 5: LLM Summary (prompt-guided on structured data) ──
    llm_summary = ""
    try:
        structured_prompt_text = _structured_to_prompt_text(medical_data)
        if structured_prompt_text:
            llm_summary = _run_llm_summary(structured_prompt_text)
            logger.info("LLM summary: %d chars", len(llm_summary))
        else:
            # No structured data — fall back to raw text summarization
            llm_summary = _run_llm_summary(cleaned_text[:2000])
            logger.info("LLM summary (raw fallback): %d chars", len(llm_summary))
    except Exception as exc:
        logger.warning("LLM summarization failed: %s", exc)

    # ── Stage 6: Render structured output ──
    summary = _render_structured_output(medical_data, llm_summary, file_name)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "summary_type": "report_summary",
        "file_name": file_name,
        "summary": summary,
        "structured_data": medical_data,
        "raw_llm_summary": llm_summary,
        "report_length": len(cleaned_text),
        "latency_ms": round(elapsed_ms, 1),
    }
