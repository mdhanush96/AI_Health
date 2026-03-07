"""
MedAI – RAG Model Loader (Singleton)
Thread-safe lazy loading for:
    1. Sentence-Transformer   (query embeddings → FAISS)
    2. FAISS index + metadata  (vector store)
    3. FLAN-T5                 (generative answering)

Every artifact is loaded ONCE per process and pinned to the resolved
CUDA / CPU device from ml_engine.config.
"""

import logging
import pickle
import threading
import time

try:
    import faiss  # type: ignore[import-untyped]
except ImportError as _faiss_err:
    raise ImportError(
        "FAISS is required but not installed. "
        "Install via: pip install faiss-cpu  (or faiss-gpu for CUDA support)"
    ) from _faiss_err

from ml_engine import config
from ml_engine.model_loader import get_device

logger = logging.getLogger("ml_engine.rag")

# ---------------------------------------------------------------------------
# Singleton state — embedding model
# ---------------------------------------------------------------------------
_embed_lock = threading.Lock()
_embed_model = None
_embed_tokenizer = None
_embed_device = None
_embed_error = None

# ---------------------------------------------------------------------------
# Singleton state — FAISS index + metadata
# ---------------------------------------------------------------------------
_faiss_lock = threading.Lock()
_faiss_index = None
_metadata = None
_faiss_error = None

# ---------------------------------------------------------------------------
# Singleton state — FLAN-T5 generator
# ---------------------------------------------------------------------------
_gen_lock = threading.Lock()
_gen_model = None
_gen_tokenizer = None
_gen_device = None
_gen_error = None


# ===================================================================
# Embedding Model  (sentence-transformers/all-MiniLM-L6-v2)
# ===================================================================

def load_embedding_model():
    """
    Load sentence-transformer for query encoding.
    Returns (model, tokenizer, device).
    """
    global _embed_model, _embed_tokenizer, _embed_device, _embed_error

    if _embed_model is not None:
        return _embed_model, _embed_tokenizer, _embed_device

    with _embed_lock:
        if _embed_model is not None:
            return _embed_model, _embed_tokenizer, _embed_device

        if _embed_error is not None:
            raise RuntimeError(_embed_error)

        try:
            from transformers import AutoModel, AutoTokenizer

            t0 = time.perf_counter()
            _embed_device = get_device()
            model_name = config.EMBEDDING_MODEL_NAME

            logger.info("Loading embedding tokenizer: %s …", model_name)
            _embed_tokenizer = AutoTokenizer.from_pretrained(model_name)

            logger.info("Loading embedding model: %s …", model_name)
            _embed_model = AutoModel.from_pretrained(model_name)
            _embed_model.to(_embed_device)
            _embed_model.eval()

            elapsed = time.perf_counter() - t0
            logger.info(
                "Embedding model loaded | %s | Device: %s | %.2fs",
                model_name, _embed_device, elapsed,
            )
            return _embed_model, _embed_tokenizer, _embed_device

        except Exception as exc:
            _embed_error = f"Failed to load embedding model: {exc}"
            logger.exception(_embed_error)
            raise RuntimeError(_embed_error) from exc


# ===================================================================
# FAISS Index + Metadata
# ===================================================================

def load_faiss_index():
    """
    Load FAISS index and metadata pickle.
    Returns (faiss_index, metadata_list).
    """
    global _faiss_index, _metadata, _faiss_error

    if _faiss_index is not None:
        return _faiss_index, _metadata

    with _faiss_lock:
        if _faiss_index is not None:
            return _faiss_index, _metadata

        if _faiss_error is not None:
            raise RuntimeError(_faiss_error)

        try:
            import os
            t0 = time.perf_counter()

            idx_path = config.FAISS_INDEX_PATH
            meta_path = config.METADATA_PATH

            if not os.path.isfile(idx_path):
                raise FileNotFoundError(f"FAISS index not found: {idx_path}")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            logger.info("Loading FAISS index from %s …", idx_path)
            _faiss_index = faiss.read_index(idx_path)

            logger.info("Loading metadata from %s …", meta_path)
            with open(meta_path, "rb") as f:
                _metadata = pickle.load(f)

            # Validate alignment
            if _faiss_index.ntotal != len(_metadata):
                raise ValueError(
                    f"FAISS vectors ({_faiss_index.ntotal}) != metadata entries ({len(_metadata)})"
                )

            elapsed = time.perf_counter() - t0
            logger.info(
                "FAISS loaded | Vectors: %d | Dim: %d | Metadata entries: %d | %.2fs",
                _faiss_index.ntotal, _faiss_index.d, len(_metadata), elapsed,
            )
            return _faiss_index, _metadata

        except Exception as exc:
            _faiss_error = f"Failed to load FAISS index: {exc}"
            logger.exception(_faiss_error)
            raise RuntimeError(_faiss_error) from exc


# ===================================================================
# FLAN-T5 Generator
# ===================================================================

def load_generator_model():
    """
    Load FLAN-T5 for context-grounded text generation.
    Returns (model, tokenizer, device).
    """
    global _gen_model, _gen_tokenizer, _gen_device, _gen_error

    if _gen_model is not None:
        return _gen_model, _gen_tokenizer, _gen_device

    with _gen_lock:
        if _gen_model is not None:
            return _gen_model, _gen_tokenizer, _gen_device

        if _gen_error is not None:
            raise RuntimeError(_gen_error)

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            t0 = time.perf_counter()
            _gen_device = get_device()
            model_name = config.GENERATOR_MODEL_NAME

            logger.info("Loading FLAN-T5 tokenizer: %s …", model_name)
            _gen_tokenizer = AutoTokenizer.from_pretrained(model_name)

            logger.info("Loading FLAN-T5 model: %s …", model_name)
            _gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _gen_model.to(_gen_device)
            _gen_model.eval()

            param_count = sum(p.numel() for p in _gen_model.parameters())
            elapsed = time.perf_counter() - t0
            logger.info(
                "FLAN-T5 loaded | Params: %s | Device: %s | %.2fs",
                f"{param_count:,}", _gen_device, elapsed,
            )
            return _gen_model, _gen_tokenizer, _gen_device

        except Exception as exc:
            _gen_error = f"Failed to load FLAN-T5: {exc}"
            logger.exception(_gen_error)
            raise RuntimeError(_gen_error) from exc


# ===================================================================
# Diagnostics
# ===================================================================

def get_rag_status() -> dict:
    """Return loading status of all RAG components."""
    return {
        "embedding_model_loaded": _embed_model is not None,
        "embedding_device": str(_embed_device) if _embed_device else None,
        "faiss_loaded": _faiss_index is not None,
        "faiss_vectors": _faiss_index.ntotal if _faiss_index else 0,
        "faiss_dimension": _faiss_index.d if _faiss_index else 0,
        "metadata_entries": len(_metadata) if _metadata else 0,
        "generator_loaded": _gen_model is not None,
        "generator_device": str(_gen_device) if _gen_device else None,
        "generator_model": config.GENERATOR_MODEL_NAME,
    }
