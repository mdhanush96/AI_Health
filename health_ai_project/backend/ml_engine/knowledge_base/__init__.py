"""
MedAI – JSON Knowledge Base Loader
Loads per-disease JSON files from the knowledge_base/ directory and provides
a simple lookup API used by the RAG pipeline and predictor.

Each JSON file has the structure:
    {
        "Overview": "...",
        "Symptoms": "...",
        "Causes": "...",
        "Treatment": "...",
        "When to See a Doctor": "...",
        "Preventions": "..."          # optional
    }
"""

import json
import logging
import os
from pathlib import Path

from ml_engine import config

logger = logging.getLogger("ml_engine.knowledge_base")

# ---------------------------------------------------------------------------
# In-memory store  (loaded once, immutable after startup)
# ---------------------------------------------------------------------------
_knowledge_base: dict[str, dict[str, str]] = {}
_loaded = False


def _normalize_name(name: str) -> str:
    """Lowercase + strip for consistent lookup keys."""
    return name.lower().strip()


def load_knowledge_base() -> dict[str, dict[str, str]]:
    """
    Load all disease JSON files from KNOWLEDGE_BASE_DIR into memory.
    Returns dict keyed by normalised disease name → section dict.
    Thread-safe: multiple calls return the same cached dict.
    """
    global _knowledge_base, _loaded

    if _loaded:
        return _knowledge_base

    kb_dir = Path(config.KNOWLEDGE_BASE_DIR)

    if not kb_dir.exists():
        logger.warning("Knowledge base directory not found: %s", kb_dir)
        _loaded = True
        return _knowledge_base

    count = 0
    for filepath in sorted(kb_dir.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data:
                logger.debug("Skipping empty file: %s", filepath.name)
                continue

            disease_name = _normalize_name(filepath.stem)
            _knowledge_base[disease_name] = data
            count += 1

        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load %s: %s", filepath.name, exc)

    _loaded = True
    logger.info(
        "Knowledge base loaded: %d diseases from %s", count, kb_dir,
    )
    return _knowledge_base


def get_disease_info(disease_name: str) -> dict[str, str] | None:
    """
    Lookup a disease by name (case-insensitive, fuzzy).

    Returns the section dict or None if not found.
    Tries exact match first, then substring match.
    """
    kb = load_knowledge_base()
    key = _normalize_name(disease_name)

    # Exact match
    if key in kb:
        return kb[key]

    # Substring / fuzzy match
    for stored_name, data in kb.items():
        if key in stored_name or stored_name in key:
            return data

    return None


def get_all_diseases() -> list[str]:
    """Return list of all disease names in the knowledge base."""
    kb = load_knowledge_base()
    return list(kb.keys())


def get_disease_sections(disease_name: str) -> dict[str, str]:
    """
    Return a normalised section dict for a disease.

    Keys are mapped to lowercase for uniform access:
        overview, symptoms, causes, treatment, when_to_see_a_doctor, preventions

    Returns empty dict if disease not found.
    """
    raw = get_disease_info(disease_name)
    if raw is None:
        return {}

    section_map = {
        "Overview": "overview",
        "Symptoms": "symptoms",
        "Causes": "causes",
        "Treatment": "treatment",
        "When to See a Doctor": "when_to_see_a_doctor",
        "Preventions": "preventions",
    }

    normalised: dict[str, str] = {}
    for orig_key, norm_key in section_map.items():
        value = raw.get(orig_key, "")
        if value and isinstance(value, str) and value.strip():
            normalised[norm_key] = value.strip()

    return normalised
