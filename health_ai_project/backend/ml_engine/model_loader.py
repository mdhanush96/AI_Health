"""
MedAI – Singleton Model Loader with GPU/CUDA Detection
Loads ClinicalBERT once and reuses across all inference requests.
Automatic GPU detection with CPU fallback.
Centralized device management for all ML components.
"""

import logging
import os
import threading
import time

from . import config

logger = logging.getLogger("ml_engine")

MODEL_PATH = config.CLINICALBERT_MODEL_PATH

# ---------------------------------------------------------------------------
# Thread-safe singleton state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_tokenizer = None
_model = None
_device = None
_load_error = None


def get_device():
    """
    Centralized CUDA device resolver.
    Respects MEDAI_FORCE_CPU and MEDAI_CUDA_DEVICE env vars via config.
    Returns torch.device.
    """
    import torch

    if config.FORCE_CPU:
        logger.info("FORCE_CPU enabled — using CPU.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        idx = config.CUDA_DEVICE_INDEX
        gpu_name = torch.cuda.get_device_name(idx)
        vram_mb = torch.cuda.get_device_properties(idx).total_mem / (1024 ** 2)
        logger.info(
            "GPU detected: %s | VRAM: %.0f MB | CUDA: %s | Device index: %d",
            gpu_name, vram_mb, torch.version.cuda, idx,
        )
        return torch.device(f"cuda:{idx}")

    logger.info("No CUDA GPU detected. Using CPU for inference.")
    return torch.device("cpu")


def _apply_precision(model):
    """Apply configured precision (float32 / float16) to model weights."""
    import torch

    if config.MODEL_PRECISION == "float16":
        model = model.half()
        logger.info("Model cast to float16 (half precision).")
    return model


def load_model_and_tokenizer():
    """
    Thread-safe singleton loader for ClinicalBERT.
    Returns (model, tokenizer, device) tuple.
    Raises RuntimeError with descriptive message on failure.
    """
    global _tokenizer, _model, _device, _load_error

    # Fast path — already loaded
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _device

    with _lock:
        # Double-check after acquiring lock
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer, _device

        if _load_error is not None:
            raise RuntimeError(_load_error)

        if not os.path.isdir(MODEL_PATH):
            _load_error = (
                f"Model directory not found: {MODEL_PATH}. "
                "Ensure the clinicalbert-disease folder is placed inside ml_engine/."
            )
            logger.error(_load_error)
            raise RuntimeError(_load_error)

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            t0 = time.perf_counter()
            _device = get_device()

            logger.info("Loading ClinicalBERT tokenizer from %s …", MODEL_PATH)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

            logger.info("Loading ClinicalBERT model from %s …", MODEL_PATH)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            _model = _apply_precision(_model)
            _model.to(_device)
            _model.eval()

            param_count = sum(p.numel() for p in _model.parameters())
            label_count = len(_model.config.id2label)
            elapsed = time.perf_counter() - t0
            logger.info(
                "ClinicalBERT loaded | Params: %s | Labels: %d | Device: %s | %.2fs",
                f"{param_count:,}", label_count, _device, elapsed,
            )

            return _model, _tokenizer, _device

        except Exception as exc:
            _load_error = f"Failed to load ClinicalBERT: {exc}"
            logger.exception(_load_error)
            raise RuntimeError(_load_error) from exc


def get_model_info() -> dict:
    """Return diagnostic info about loaded ClinicalBERT model."""
    if _model is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "device": str(_device),
        "parameters": sum(p.numel() for p in _model.parameters()),
        "labels": len(_model.config.id2label),
        "precision": config.MODEL_PRECISION,
    }