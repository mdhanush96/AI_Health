"""
MedAI – Centralized ML Configuration
All model paths, hyperparameters, and device settings in one place.
Config-driven architecture for easy upgrades and environment overrides.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
ML_ENGINE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ML_ENGINE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent.parent  # health_ai_project/../

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------
# Force CPU via env var: MEDAI_FORCE_CPU=1
FORCE_CPU = os.environ.get("MEDAI_FORCE_CPU", "0").lower() in ("1", "true", "yes")

# CUDA device index (multi-GPU support)
CUDA_DEVICE_INDEX = int(os.environ.get("MEDAI_CUDA_DEVICE", "0"))

# ---------------------------------------------------------------------------
# ClinicalBERT Configuration
# ---------------------------------------------------------------------------
CLINICALBERT_MODEL_PATH = str(ML_ENGINE_DIR / "clinicalbert-disease")
CLINICALBERT_MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Sentence Transformer (Retriever Embeddings)
# ---------------------------------------------------------------------------
# Model used to build the FAISS index — must match at query time
EMBEDDING_MODEL_NAME = os.environ.get(
    "MEDAI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dim

# ---------------------------------------------------------------------------
# FAISS Retriever Configuration
# ---------------------------------------------------------------------------
RAG_DIR = ML_ENGINE_DIR / "rag"
FAISS_INDEX_PATH = str(RAG_DIR / "faiss_index.bin")
METADATA_PATH = str(RAG_DIR / "metadata.pkl")

# Retrieval defaults
RETRIEVER_TOP_K = int(os.environ.get("MEDAI_RETRIEVER_TOP_K", "10"))
RETRIEVER_SCORE_THRESHOLD = float(os.environ.get("MEDAI_RETRIEVER_THRESHOLD", "2.0"))

# ---------------------------------------------------------------------------
# FLAN-T5 Generator Configuration
# ---------------------------------------------------------------------------
GENERATOR_MODEL_NAME = os.environ.get(
    "MEDAI_GENERATOR_MODEL", "google/flan-t5-base"
)
GENERATOR_MAX_INPUT_TOKENS = 1024
GENERATOR_MAX_OUTPUT_TOKENS = 512
GENERATOR_TEMPERATURE = 0.7
GENERATOR_NUM_BEAMS = 4
GENERATOR_DO_SAMPLE = False
GENERATOR_TOP_P = 0.9

# ---------------------------------------------------------------------------
# RAG Pipeline Configuration
# ---------------------------------------------------------------------------
RAG_ENABLED = os.environ.get("MEDAI_RAG_ENABLED", "1").lower() in ("1", "true", "yes")

# Cache TTL for RAG responses (seconds). 0 = disabled.
RAG_CACHE_TTL = int(os.environ.get("MEDAI_RAG_CACHE_TTL", "300"))

# ---------------------------------------------------------------------------
# Disease Knowledge Base
# ---------------------------------------------------------------------------
DISEASE_INFO_PATH = str(ML_ENGINE_DIR / "disease_info.json")

# JSON Knowledge Base — one JSON file per disease (Overview, Symptoms, etc.)
KNOWLEDGE_BASE_DIR = ML_ENGINE_DIR / "knowledge_base"

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
# Float precision: "float32" or "float16" (half precision for lower VRAM)
MODEL_PRECISION = os.environ.get("MEDAI_MODEL_PRECISION", "float32")

# Preload models at Django startup (AppConfig.ready())
PRELOAD_MODELS = os.environ.get("MEDAI_PRELOAD", "0").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_INFERENCE_INPUTS = os.environ.get("MEDAI_LOG_INPUTS", "1").lower() in ("1", "true", "yes")
LOG_INFERENCE_LATENCY = True
