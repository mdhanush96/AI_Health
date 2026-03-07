"""
MedAI – Hybrid RAG Pipeline Orchestrator
Combines FAISS retrieval + FLAN-T5 generation into a single callable.
Optionally incorporates ClinicalBERT classification results for
disease-targeted retrieval.

Knowledge sources:
  1. JSON knowledge base (knowledge_base/*.json) — structured per-disease data
  2. FAISS vector index (faiss_index.bin) + metadata (metadata.pkl) — fallback
"""

import hashlib
import logging
import re
import threading
import time

from ml_engine import config
from ml_engine.knowledge_base import get_disease_sections
from .retriever import retrieve, retrieve_all_for_disease, retrieve_for_disease
from .generator import generate

logger = logging.getLogger("ml_engine.rag")

# ---------------------------------------------------------------------------
# In-memory response cache (thread-safe)
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_response_cache: dict[str, tuple[float, dict]] = {}  # key → (timestamp, response)
MAX_CACHE_SIZE = 200


def _cache_key(query: str, disease: str | None) -> str:
    raw = f"{query.strip().lower()}||{(disease or '').lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _get_cached(key: str) -> dict | None:
    if config.RAG_CACHE_TTL <= 0:
        return None
    with _cache_lock:
        entry = _response_cache.get(key)
        if entry is None:
            return None
        ts, resp = entry
        if (time.time() - ts) > config.RAG_CACHE_TTL:
            del _response_cache[key]
            return None
        return resp


def _set_cached(key: str, response: dict):
    if config.RAG_CACHE_TTL <= 0:
        return
    with _cache_lock:
        if len(_response_cache) >= MAX_CACHE_SIZE:
            oldest_key = min(_response_cache, key=lambda k: _response_cache[k][0])
            del _response_cache[oldest_key]
        _response_cache[key] = (time.time(), response)


# ---------------------------------------------------------------------------
# Text Sanitization
# ---------------------------------------------------------------------------

def _disease_match(chunk_disease: str, target_disease: str) -> bool:
    """Fuzzy disease name matching (handles metadata variants like 'GERD' vs full name)."""
    a = chunk_disease.lower().strip()
    b = target_disease.lower().strip()
    return a == b or b in a or a in b


def _sanitize_text(text: str) -> str:
    """
    Remove citation noise, URLs, markdown links, and other artifacts
    from retrieved chunk content or generated text.
    """
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)   # [text](url) → text
    text = re.sub(r'https?://\S+', '', text)                # bare URLs
    text = re.sub(r'\[\d+\]?', '', text)                    # [1] or unclosed [116
    text = re.sub(r'\(\d+\)', '', text)                     # (1), (23)
    text = re.sub(r'citation:\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)         # **bold**
    text = re.sub(r'(?<!\w)\*\s', ' ', text)                # * bullet points
    text = re.sub(r'\*([^*]+)\*', r'\1', text)              # *italic*
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)  # full header lines
    text = re.sub(r'#{1,6}\s+', '', text)                    # inline ### markers
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)             # [stray brackets] → content
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'[\[\]]', '', text)                       # any remaining [ or ]
    text = re.sub(r'##\s*', '', text)                        # leftover ##
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'^[,;.\s]+', '', text)
    if text and text[-1] not in '.!?':
        text += '.'
    return text


# ---------------------------------------------------------------------------
# Chunk-based extraction helpers
# ---------------------------------------------------------------------------

# Comprehensive symptom vocabulary for extraction from chunk text
_SYMPTOM_KEYWORDS = [
    # General
    "fever", "high fever", "fatigue", "weakness", "malaise", "lethargy",
    "weight loss", "weight gain", "chills", "sweating", "night sweats",
    # Head / neurological
    "headache", "severe headache", "dizziness", "confusion", "seizures",
    "blurred vision", "light sensitivity", "migraine",
    # Respiratory
    "cough", "wheezing", "shortness of breath", "breathlessness",
    "chest tightness", "sore throat", "runny nose", "sneezing",
    # Cardiovascular
    "chest pain", "palpitations", "rapid heartbeat",
    # GI
    "nausea", "vomiting", "diarrhea", "bloating", "abdominal pain",
    "stomach pain", "heartburn", "acid reflux", "burning sensation",
    "loss of appetite", "indigestion",
    # Musculoskeletal
    "joint pain", "muscle pain", "stiffness", "swelling", "arthralgia",
    "myalgia", "back pain", "neck pain",
    # Skin
    "rash", "skin rash", "itching", "red patches", "blisters",
    "red sores", "scaly skin", "hives", "bruising",
    # Urinary
    "frequent urination", "painful urination", "burning urination",
    "blood in urine", "dysuria",
    # Other
    "bleeding", "numbness", "tingling", "cramps",
    "yellow skin", "dark urine", "swollen lymph nodes",
]


def _extract_symptoms_from_chunks(
    chunks: list[dict],
    query: str,
) -> list[str]:
    """
    Extract symptom mentions from retrieved FAISS chunk content.
    Prioritises symptoms that appear in both the chunks AND the user query.
    """
    combined_text = " ".join(c["content"].lower() for c in chunks[:5])
    query_lower = query.lower()

    in_both: list[str] = []
    in_chunks_only: list[str] = []

    for kw in _SYMPTOM_KEYWORDS:
        if kw in combined_text:
            if kw in query_lower:
                in_both.append(kw.title())
            else:
                in_chunks_only.append(kw.title())

    # Deduplicate substrings (e.g. keep "Severe Headache" but drop "Headache")
    all_found = in_both + in_chunks_only
    deduped: list[str] = []
    for s in all_found:
        if not any(s != other and s in other for other in all_found):
            deduped.append(s)

    return deduped[:10]


# ---------------------------------------------------------------------------
# Structured Section Extraction from Raw Chunks
# ---------------------------------------------------------------------------

_SECTION_MAPPING = {
    "overview": "description",
    "description": "description",
    "introduction": "description",
    "causes": "causes",
    "cause": "causes",
    "risk factors": "causes",
    "etiology": "causes",
    "pathophysiology": "causes",
    "symptoms": "symptoms",
    "symptom": "symptoms",
    "clinical features": "symptoms",
    "signs and symptoms": "symptoms",
    "complications": "complications",
    "complication": "complications",
    "possible complications": "complications",
    "preventions": "recommendations",
    "prevention": "recommendations",
    "lifestyle modifications": "recommendations",
    "lifestyle": "recommendations",
    "dietary": "recommendations",
    "treatment": "treatment",
    "management": "treatment",
    "diagnosis": "diagnosis",
    "when to see a doctor": "emergency",
    "when to seek": "emergency",
    "emergency": "emergency",
    "seek medical": "emergency",
}


def _classify_section_header(header: str) -> str:
    """Map a markdown header text to a section category."""
    h = header.lower().strip()
    for keyword, category in _SECTION_MAPPING.items():
        if keyword in h:
            return category
    return "other"


def _normalize_chunk_text(text: str) -> str:
    """
    Insert newlines before markdown structures.
    FAISS chunks are stored as single-line strings — this enables
    line-based section and bullet parsing.
    """
    # Add newline before ### headers
    text = re.sub(r'\s*(#{1,4}\s+)', r'\n\1', text)
    # Add newline before markdown bullet items: * **text**
    text = re.sub(r'\s*(\* \*\*)', r'\n\1', text)
    # Add newline before plain bullets: * Capitalized text (but not mid-sentence)
    text = re.sub(r'(?<=[.!?:])\s*(\* [A-Z])', r'\n\1', text)
    return text.strip()


def _parse_disease_sections(chunks: list[dict]) -> dict[str, str]:
    """
    Concatenate all FAISS chunks for a disease, normalize newlines,
    and parse into named sections by splitting on markdown ### headers.

    Returns a dict mapping section category names to their raw content.
    Text before the first header goes under 'description'.
    """
    # Sort chunks by FAISS index to reconstruct document order
    sorted_chunks = sorted(chunks, key=lambda c: c.get("faiss_idx", 0))

    # Space-join in document order, then normalize.
    # Chunks are sequential passages from the same document — space-join
    # reconnects text that was split at chunk boundaries.
    combined = _normalize_chunk_text(" ".join(c["content"] for c in sorted_chunks))

    sections: dict[str, list[str]] = {}
    current_section = "description"
    current_lines: list[str] = []

    for line in combined.split('\n'):
        header_match = re.match(r'^#{1,4}\s+(.+)$', line.strip())
        if header_match:
            # Save content accumulated for previous section
            content = '\n'.join(current_lines).strip()
            if content:
                sections.setdefault(current_section, []).append(content)

            full_text = header_match.group(1)
            full_lower = full_text.lower().strip()

            # Find the longest matching section keyword at start of header
            best_kw_len = 0
            best_category = "other"
            for keyword, category in _SECTION_MAPPING.items():
                if full_lower.startswith(keyword) and len(keyword) > best_kw_len:
                    best_kw_len = len(keyword)
                    best_category = category

            # Fallback: check if any keyword is contained anywhere
            if best_kw_len == 0:
                best_category = _classify_section_header(full_text)

            current_section = best_category
            # Everything after the matched keyword is section content
            if best_kw_len > 0:
                leftover = full_text[best_kw_len:].strip()
            else:
                leftover = ""
            current_lines = [leftover] if leftover else []
        else:
            current_lines.append(line)

    content = '\n'.join(current_lines).strip()
    if content:
        sections.setdefault(current_section, []).append(content)

    return {k: '\n\n'.join(v) for k, v in sections.items()}


def _clean_sentences(
    raw_text: str,
    max_sentences: int = 4,
    max_chars: int = 600,
) -> str:
    """Sanitize text and extract first N well-formed sentences."""
    text = _sanitize_text(raw_text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    result: list[str] = []
    total_len = 0
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 15:
            continue
        if not s[0].isupper():
            continue
        result.append(s)
        total_len += len(s)
        if len(result) >= max_sentences or total_len >= max_chars:
            break

    return ' '.join(result) if result else ''


def _extract_bullet_items(
    raw_text: str,
    max_items: int = 8,
    max_item_len: int = 250,
) -> list[str]:
    """
    Extract bullet point items from text (line-based + inline fallback).
    Handles: * **Bold:** description  |  * plain text  |  - text
    Works on both multi-line and single-line input.
    """
    # Normalize: insert newlines before inline bullets if not present
    normalized = _normalize_chunk_text(raw_text)

    items: list[str] = []
    for line in normalized.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Match: * **Label:** description
        m = re.match(r'^\*\s+\*\*(.+?)\*\*\s*:?\s*(.*)', line)
        if m:
            label = _sanitize_text(m.group(1).strip()).rstrip('.').rstrip(':')
            desc = m.group(2).strip()
            if desc:
                desc = _sanitize_text(desc).rstrip('.')
                if len(desc) > max_item_len:
                    cut = desc[:max_item_len]
                    lp = cut.rfind('.')
                    desc = cut[:lp + 1] if lp > 60 else cut + '…'
                items.append(f"{label} — {desc}")
            else:
                # Skip label-only items that are just sub-headers
                if label and len(label) > 10 and not any(
                    kw in label.lower()
                    for kw in ("modifications", "adjustments", "overview", "general")
                ):
                    items.append(label)
        elif re.match(r'^[*\-]\s+', line):
            # Plain bullet
            text = re.sub(r'^[*\-]\s+', '', line)
            text = _sanitize_text(text).rstrip('.')
            if text and len(text) > 10:
                if len(text) > max_item_len:
                    cut = text[:max_item_len]
                    lp = cut.rfind('.')
                    text = cut[:lp + 1] if lp > 60 else cut + '…'
                items.append(text)
        if len(items) >= max_items:
            break

    return items


def _get_description(sections: dict[str, str], disease_name: str) -> str:
    """Extract a clean 2-4 sentence disease description."""
    desc_raw = sections.get("description", "")
    if desc_raw:
        result = _clean_sentences(desc_raw, max_sentences=4, max_chars=600)
        if result:
            return result

    # Fallback: use any available section content
    for section in ["other", "symptoms", "causes"]:
        if section in sections:
            result = _clean_sentences(sections[section], max_sentences=2, max_chars=300)
            if result:
                return result

    return (
        f"{disease_name.title()} is a medical condition. "
        "Please consult a healthcare professional for detailed information."
    )


def _get_causes(sections: dict[str, str], combined_raw: str) -> list[str]:
    """Extract causes and risk factors as a list of items."""
    causes_raw = sections.get("causes", "")

    if causes_raw:
        items = _extract_bullet_items(causes_raw, max_items=6)
        if items:
            return items
        text = _clean_sentences(causes_raw, max_sentences=3, max_chars=400)
        if text:
            return [text]

    # Fallback: search for cause-related sentences in all text
    cause_keywords = [
        "caused by", "due to", "risk factor", "occurs when",
        "results from", "contributes to", "triggered by",
    ]
    # Exclude sentences that are clearly recommendations/prevention
    exclude_keywords = [
        "should", "avoid", "management", "achieving", "maintain",
        "recommended", "prevention", "consult",
    ]
    text = _sanitize_text(combined_raw)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    found: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20 or len(s) > 250 or not s[0].isupper():
            continue
        s_lower = s.lower()
        if any(kw in s_lower for kw in cause_keywords):
            if not any(ex in s_lower for ex in exclude_keywords):
                found.append(s)
        if len(found) >= 3:
            break

    return found if found else [
        "Causes may vary depending on individual factors. "
        "Consult a healthcare professional for a thorough evaluation."
    ]


def _get_symptoms_detailed(
    chunks: list[dict],
    query: str,
    sections: dict[str, str],
) -> list[str]:
    """
    Extract symptoms combining:
    1. Bullet items from parsed Symptoms section
    2. Symptom lists from prose (e.g., "symptoms include X, Y, and Z")
    3. Keyword-based extraction from full chunk text
    Returns de-duplicated list suitable for bullet-point display.
    """
    # First: try bullet items from the Symptoms section
    symptom_section = sections.get("symptoms", "")
    section_bullets = (
        _extract_bullet_items(symptom_section, max_items=10, max_item_len=120)
        if symptom_section else []
    )

    # Second: try to extract symptom lists from prose text
    prose_symptoms: list[str] = []
    if symptom_section:
        text = _sanitize_text(symptom_section)
        # Match patterns like "symptoms include heartburn, regurgitation, and pain"
        list_match = re.search(
            r'(?:symptoms?|manifestations?|features?)\s+(?:include|are|such as|like)\s+([^.]+)',
            text, re.IGNORECASE,
        )
        if list_match:
            items_text = list_match.group(1)
            # Split by comma and 'and'
            for item in re.split(r',\s*|\s+and\s+', items_text):
                item = item.strip().rstrip('.')
                # Strip leading conjunctions/articles
                item = re.sub(r'^(?:and|or|the|a|an)\s+', '', item, flags=re.IGNORECASE)
                if item and len(item) > 3 and len(item) < 50:
                    prose_symptoms.append(item.title())

    # Third: keyword-based extraction
    keyword_symptoms = _extract_symptoms_from_chunks(chunks, query)

    # Merge: section bullets \u2192 prose \u2192 keywords (dedup by fuzzy match)
    seen_lower: set[str] = set()
    merged: list[str] = []
    for s in section_bullets + prose_symptoms + keyword_symptoms:
        key = s.lower().strip()
        # Fuzzy dedup: skip if any existing item contains this as substring or vice-versa
        if key in seen_lower or any(
            key in existing or existing in key
            for existing in seen_lower
        ):
            continue
        seen_lower.add(key)
        merged.append(s)

    return merged[:12]


def _get_complications(sections: dict[str, str], combined_raw: str) -> str:
    """Extract possible complications text."""
    comp_raw = sections.get("complications", "")

    if comp_raw:
        result = _clean_sentences(comp_raw, max_sentences=3, max_chars=350)
        if result:
            return result

    # Fallback: search for complication-specific sentences
    comp_keywords = [
        "may lead to", "can progress", "can cause serious",
        "if left untreated", "long-term damage",
        "develop into", "associated with a high risk",
        "can lead to", "severe form", "life-threatening",
    ]
    # Exclude sentences that are about prevention/lifestyle
    comp_exclude = [
        "smoking cessation", "weight management", "avoid",
        "prevention", "recommended", "should",
    ]
    text = _sanitize_text(combined_raw)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for s in sentences:
        s = s.strip()
        if len(s) < 25 or len(s) > 300 or not s[0].isupper():
            continue
        s_lower = s.lower()
        if any(kw in s_lower for kw in comp_keywords):
            if not any(ex in s_lower for ex in comp_exclude):
                return s

    return (
        "If left untreated, this condition may worsen over time. "
        "Early medical evaluation is recommended."
    )


def _get_recommendations(sections: dict[str, str]) -> list[str]:
    """Extract lifestyle and dietary recommendations."""
    rec_raw = sections.get("recommendations", "") or sections.get("treatment", "")

    if rec_raw:
        items = _extract_bullet_items(rec_raw, max_items=6)
        if items:
            return items
        text = _clean_sentences(rec_raw, max_sentences=4, max_chars=500)
        if text:
            return [text]

    return [
        "Follow your doctor's prescribed treatment plan",
        "Maintain a balanced diet and adequate hydration",
        "Get sufficient rest and manage stress levels",
    ]


# ---------------------------------------------------------------------------
# Response Formatting
# ---------------------------------------------------------------------------

def _build_disease_section_from_kb(
    disease_name: str,
    kb_sections: dict[str, str],
    rank: int,
    confidence: float | None = None,
    include_full_detail: bool = False,
) -> list[str]:
    """
    Build a structured report section for a disease directly from the
    JSON knowledge base (no FAISS chunks needed).
    """
    lines: list[str] = []
    display = disease_name.title()
    conf_str = f" ({confidence:.1f}%)" if confidence is not None else ""

    if rank == 1:
        lines.append(f"\U0001f3e5 Likely Condition: {display}")
    else:
        lines.append(f"\U0001f3e5 {rank}. {display}{conf_str}")
    lines.append("")

    # Overview / Description
    overview = kb_sections.get("overview", "")
    if overview:
        lines.append(_sanitize_text(overview))
        lines.append("")
    else:
        lines.append(
            f"{display} — limited information available in the knowledge base."
        )
        lines.append("")

    if include_full_detail:
        # Causes
        causes = kb_sections.get("causes", "")
        if causes:
            lines.append("Causes:")
            lines.append(f"  {_sanitize_text(causes)}")
            lines.append("")

        # Symptoms
        symptoms = kb_sections.get("symptoms", "")
        if symptoms:
            lines.append("Common Symptoms:")
            lines.append(f"  {_sanitize_text(symptoms)}")
            lines.append("")

        # Treatment
        treatment = kb_sections.get("treatment", "")
        if treatment:
            lines.append("Treatment:")
            lines.append(f"  {_sanitize_text(treatment)}")
            lines.append("")

        # Preventions / Recommendations
        preventions = kb_sections.get("preventions", "")
        if preventions:
            lines.append("Lifestyle & Dietary Recommendations:")
            lines.append(f"  {_sanitize_text(preventions)}")
            lines.append("")

        # When to See a Doctor
        when_doc = kb_sections.get("when_to_see_a_doctor", "")
        lines.append("When to Seek Medical Attention:")
        if when_doc:
            lines.append(f"  {_sanitize_text(when_doc)}")
        else:
            lines.append(
                "  Seek immediate care if symptoms worsen significantly, you experience"
            )
            lines.append(
                "  difficulty breathing, chest pain, or high fever persisting beyond 3 days."
            )
        lines.append("")
    else:
        # Brief format for secondary predictions
        symptoms = kb_sections.get("symptoms", "")
        if symptoms:
            lines.append("Common Symptoms:")
            lines.append(f"  {_sanitize_text(symptoms[:300])}")
            lines.append("")

        treatment = kb_sections.get("treatment", "")
        if treatment:
            lines.append(f"\U0001f4a1 {_sanitize_text(treatment[:200])}")
        else:
            lines.append("\U0001f4a1 Consult a healthcare professional for proper evaluation.")
        lines.append("")

    return lines


def _build_disease_section(
    disease_name: str,
    chunks: list[dict],
    query: str,
    rank: int,
    confidence: float | None = None,
    include_full_detail: bool = False,
) -> list[str]:
    """
    Build a structured report section for a single disease.

    Prefers the JSON knowledge base when available.
    Falls back to FAISS chunk parsing otherwise.

    For rank-1 (top prediction) with include_full_detail=True:
        Full description, causes, symptoms, complications, recommendations,
        and when-to-seek-care section.

    For secondary predictions:
        Brief description + symptoms + one-line recommendation.
    """
    # --- Try JSON knowledge base first ---
    kb_sections = get_disease_sections(disease_name)
    if kb_sections:
        return _build_disease_section_from_kb(
            disease_name, kb_sections, rank, confidence, include_full_detail,
        )
    lines: list[str] = []
    display = disease_name.title()
    conf_str = f" ({confidence:.1f}%)" if confidence is not None else ""

    # --- Header ---
    if rank == 1:
        lines.append(f"🏥 Likely Condition: {display}")
    else:
        lines.append(f"🏥 {rank}. {display}{conf_str}")
    lines.append("")

    if not chunks:
        lines.append(
            f"{display} — limited information available in the knowledge base."
        )
        lines.append("")
        lines.append("💡 Consult a healthcare professional for proper evaluation.")
        lines.append("")
        return lines

    # Parse all chunks into structured sections
    sections = _parse_disease_sections(chunks)
    combined_raw = "\n\n".join(c["content"] for c in chunks)

    # --- Description (2-4 sentences) ---
    description = _get_description(sections, disease_name)
    lines.append(description)
    lines.append("")

    if include_full_detail:
        # --- Causes ---
        causes = _get_causes(sections, combined_raw)
        if causes:
            lines.append("Causes:")
            for cause in causes:
                lines.append(f"  • {cause}")
            lines.append("")

        # --- Common Symptoms (bullet list) ---
        symptoms = _get_symptoms_detailed(chunks, query, sections)
        if symptoms:
            lines.append("Common Symptoms:")
            for symptom in symptoms:
                lines.append(f"  • {symptom}")
            lines.append("")

        # --- Possible Complications ---
        complications = _get_complications(sections, combined_raw)
        if complications:
            lines.append("Possible Complications:")
            lines.append(f"  {complications}")
            lines.append("")

        # --- Lifestyle & Dietary Recommendations ---
        recommendations = _get_recommendations(sections)
        if recommendations:
            lines.append("Lifestyle & Dietary Recommendations:")
            for rec in recommendations:
                lines.append(f"  • {rec}")
            lines.append("")

        # --- When to Seek Medical Attention ---
        lines.append("When to Seek Medical Attention:")
        emergency_raw = sections.get("emergency", "")
        if emergency_raw:
            emergency_bullets = _extract_bullet_items(emergency_raw, max_items=4)
            if emergency_bullets:
                for eb in emergency_bullets:
                    lines.append(f"  • {eb}")
            else:
                emergency_text = _clean_sentences(emergency_raw, max_sentences=2, max_chars=300)
                if emergency_text:
                    lines.append(f"  {emergency_text}")
                else:
                    lines.append(
                        "  Seek immediate care if symptoms worsen significantly."
                    )
        else:
            lines.append(
                "  Seek immediate care if symptoms worsen significantly, you experience"
            )
            lines.append(
                "  difficulty breathing, chest pain, or high fever persisting beyond 3 days."
            )
        lines.append("")

    else:
        # --- Brief format for secondary predictions ---
        symptoms = _extract_symptoms_from_chunks(chunks, query)
        if symptoms:
            lines.append("Common Symptoms:")
            for symptom in symptoms[:6]:
                lines.append(f"  • {symptom}")
            lines.append("")

        recommendations = _get_recommendations(sections)
        if recommendations:
            lines.append(f"💡 {recommendations[0]}")
        else:
            lines.append("💡 Consult a healthcare professional for proper evaluation.")
        lines.append("")

    return lines


def _format_rag_response(
    generated_text: str,
    all_predictions: list[dict],
    retrieved_chunks: list[dict],
    query: str,
) -> str:
    """
    Build a comprehensive, structured medical report.

    Flow:
        1. Show possible conditions (all top predictions)
        2. Identify best symptom match
        3. Full detail for best match from knowledge base
        4. Brief summaries for other conditions
    """
    parts: list[str] = []

    if not all_predictions:
        parts.append(generated_text)
        return "\n".join(parts)

    # --- Possible Conditions Header ---
    parts.append("Possible conditions:")
    for i, pred in enumerate(all_predictions):
        disease = pred["disease"].title()
        conf = pred.get("confidence")
        sym_score = pred.get("symptom_match_score")
        conf_str = f" — {conf:.1f}%" if conf is not None else ""
        match_str = f" (symptom match: {sym_score:.0%})" if sym_score is not None else ""
        parts.append(f"  {i + 1}. {disease}{conf_str}{match_str}")
    parts.append("")

    # --- Best Match Identification ---
    top = all_predictions[0]
    top_disease = top["disease"]
    is_strong = top.get("symptom_match_score", 0) >= 0.15

    if is_strong:
        parts.append(
            f"The symptoms most closely match {top_disease.title()}."
        )
    else:
        parts.append(
            "No single condition is a strong symptom match. "
            "The following conditions are possible — please consult a healthcare professional."
        )
    parts.append("")
    parts.append("━" * 40)
    parts.append("")

    # --- Top Prediction: full detailed section ---
    top_conf = top.get("confidence")

    try:
        top_chunks = retrieve_all_for_disease(query, top_disease)
    except Exception:
        top_chunks = [
            c for c in retrieved_chunks
            if _disease_match(c["disease"], top_disease)
        ]
    if not top_chunks:
        top_chunks = retrieved_chunks[:3]

    parts.extend(_build_disease_section(
        top_disease, top_chunks, query,
        rank=1, confidence=top_conf, include_full_detail=True,
    ))

    # --- Other Possible Conditions ---
    other_preds = all_predictions[1:] if len(all_predictions) > 1 else []
    if other_preds:
        parts.append("━" * 40)
        parts.append("🔎 Other Possible Conditions:")
        parts.append("")

        for i, pred in enumerate(other_preds):
            disease = pred["disease"]
            conf = pred.get("confidence")

            disease_chunks = [
                c for c in retrieved_chunks
                if _disease_match(c["disease"], disease)
            ]
            if not disease_chunks:
                try:
                    disease_chunks = retrieve_for_disease(query, disease, top_k=3)
                except Exception:
                    disease_chunks = []

            parts.extend(_build_disease_section(
                disease, disease_chunks, query,
                rank=i + 2, confidence=conf, include_full_detail=False,
            ))

    # --- Disclaimer ---
    parts.append("━" * 40)
    parts.append(
        "⚠️ This system provides informational guidance only. "
        "It is NOT a medical diagnosis. Always consult a qualified "
        "healthcare professional for proper evaluation and treatment."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main RAG Query Interface
# ---------------------------------------------------------------------------

def rag_query(
    query: str,
    classified_disease: str | None = None,
    all_predictions: list[dict] | None = None,
    top_k: int | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Execute the full Hybrid RAG pipeline:
        1. Retrieve relevant chunks from FAISS
        2. Optionally filter by classified disease
        3. Generate response with FLAN-T5
        4. Build multi-disease report for all predictions

    Args:
        query: User symptom text.
        classified_disease: Top ClinicalBERT prediction (enables targeted retrieval).
        all_predictions: Full list of ClinicalBERT predictions for multi-disease report.
        top_k: Override retriever top_k.
        use_cache: Whether to use response cache.

    Returns:
        {
            "rag_response": str,
            "retrieved_chunks": [...],
            "generation_meta": {...},
            "pipeline_latency_ms": float,
            "cache_hit": bool
        }
    """
    t0 = time.perf_counter()
    top_k = top_k or config.RETRIEVER_TOP_K

    # Build predictions list if not provided
    if all_predictions is None and classified_disease:
        all_predictions = [{"disease": classified_disease}]
    elif all_predictions is None:
        all_predictions = []

    # --- Cache lookup ---
    ckey = _cache_key(query, classified_disease)
    if use_cache:
        cached = _get_cached(ckey)
        if cached is not None:
            cached["cache_hit"] = True
            logger.info("RAG cache hit for query: '%s'", query[:60])
            return cached

    # --- Step 1: Retrieve ---
    if classified_disease:
        # Hybrid: disease-targeted retrieval + general retrieval
        targeted = retrieve_for_disease(query, classified_disease, top_k=5)
        general = retrieve(query, top_k=top_k)

        # Merge: targeted first, then fill with general (deduplicated)
        seen_contents = {c["content"][:100] for c in targeted}
        merged = list(targeted)
        for chunk in general:
            if chunk["content"][:100] not in seen_contents:
                merged.append(chunk)
                seen_contents.add(chunk["content"][:100])
            if len(merged) >= top_k:
                break
        chunks = merged[:top_k]
    else:
        chunks = retrieve(query, top_k=top_k)

    if not chunks:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "rag_response": (
                "I could not find relevant medical information for your query. "
                "Please try describing your symptoms in more detail."
            ),
            "retrieved_chunks": [],
            "generation_meta": {},
            "pipeline_latency_ms": round(elapsed, 1),
            "cache_hit": False,
        }

    # --- Step 2: Generate ---
    # Pass FULL chunk content (not snippets) so the generator has maximum context
    gen_result = generate(
        query=query,
        retrieved_chunks=chunks,
        classified_disease=classified_disease,
    )

    # --- Build response ---
    # Format the generated text with multi-disease structure
    formatted_response = _format_rag_response(
        gen_result["generated_text"],
        all_predictions,
        chunks,
        query,
    )

    # Summarize chunks for the response (don't send full content to frontend)
    chunk_summaries = [
        {
            "disease": c["disease"],
            "score": c["score"],
            "rank": c["rank"],
            "snippet": c["content"][:200] + "…" if len(c["content"]) > 200 else c["content"],
        }
        for c in chunks
    ]

    pipeline_ms = (time.perf_counter() - t0) * 1000

    response = {
        "rag_response": formatted_response,
        "retrieved_chunks": chunk_summaries,
        "generation_meta": {
            "prompt_tokens": gen_result["prompt_tokens"],
            "output_tokens": gen_result["output_tokens"],
            "generation_latency_ms": gen_result["latency_ms"],
        },
        "pipeline_latency_ms": round(pipeline_ms, 1),
        "cache_hit": False,
    }

    logger.info(
        "RAG pipeline complete | Chunks: %d | Gen tokens: %d | Total: %.1fms",
        len(chunks), gen_result["output_tokens"], pipeline_ms,
    )

    # --- Cache store ---
    if use_cache:
        _set_cached(ckey, response)

    return response
