"""
MedAI – FAISS Semantic Retriever
Encodes user queries with sentence-transformer, searches FAISS index,
returns top-k relevant medical knowledge chunks with scores.
"""

import logging
import time

import numpy as np
import torch

from ml_engine import config
from .rag_loader import load_embedding_model, load_faiss_index

logger = logging.getLogger("ml_engine.rag")


def _mean_pooling(model_output, attention_mask):
    """
    Mean pooling over token embeddings, masked by attention.
    Matches the sentence-transformers default pooling strategy.
    """
    token_embeddings = model_output.last_hidden_state  # (batch, seq, dim)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single text query into a 384-dim embedding vector.
    Returns shape (1, 384) float32 numpy array ready for FAISS.
    """
    model, tokenizer, device = load_embedding_model()

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    embedding = _mean_pooling(output, inputs["attention_mask"])

    # L2 normalize (consistent with sentence-transformers)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy().astype(np.float32)


def retrieve(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Retrieve top-k relevant medical knowledge chunks for a query.

    Args:
        query: Natural language symptom description.
        top_k: Number of results (default from config).
        score_threshold: Max L2 distance to include (lower = more similar).

    Returns:
        List of dicts: {disease, content, score, rank}
        Sorted by relevance (lowest distance first).
    """
    top_k = top_k or config.RETRIEVER_TOP_K
    score_threshold = score_threshold or config.RETRIEVER_SCORE_THRESHOLD

    t0 = time.perf_counter()

    # Encode query
    query_vector = encode_query(query)

    # Search FAISS
    index, metadata = load_faiss_index()
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue  # FAISS returns -1 for empty slots
        if dist > score_threshold:
            continue  # Filter low-relevance results

        meta = metadata[idx]
        results.append({
            "disease": meta["disease"],
            "content": meta["content"],
            "score": round(float(dist), 4),
            "rank": rank + 1,
            "faiss_idx": int(idx),
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if config.LOG_INFERENCE_INPUTS:
        logger.info(
            "Retrieval complete | Query: '%s' | Results: %d/%d | %.1fms",
            query[:80], len(results), top_k, elapsed_ms,
        )

    return results


def _disease_match(chunk_disease: str, target_disease: str) -> bool:
    """Fuzzy disease name matching (handles metadata variants like 'GERD' vs full name)."""
    a = chunk_disease.lower().strip()
    b = target_disease.lower().strip()
    return a == b or b in a or a in b


def retrieve_for_disease(
    query: str,
    disease_name: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve chunks filtered to a specific disease.
    Useful for targeted context after classification.
    """
    # Get more candidates and filter
    all_results = retrieve(query, top_k=top_k * 3, score_threshold=999.0)
    filtered = [
        r for r in all_results
        if _disease_match(r["disease"], disease_name)
    ]
    return filtered[:top_k]


def retrieve_all_for_disease(
    query: str,
    disease_name: str,
) -> list[dict]:
    """
    Retrieve ALL chunks for a specific disease from the FAISS index.
    Returns all matching chunks sorted by FAISS index (document order).
    Used for building comprehensive disease reports.
    """
    index, metadata = load_faiss_index()
    query_vector = encode_query(query)

    # Search ALL vectors to find every chunk for this disease
    n_vectors = index.ntotal
    distances, indices = index.search(query_vector, n_vectors)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        if _disease_match(meta["disease"], disease_name):
            results.append({
                "disease": meta["disease"],
                "content": meta["content"],
                "score": round(float(dist), 4),
                "rank": 0,
                "faiss_idx": int(idx),
            })

    # Sort by FAISS index for document order
    results.sort(key=lambda r: r["faiss_idx"])
    for i, r in enumerate(results):
        r["rank"] = i + 1

    logger.info(
        "Full disease retrieval | Disease: '%s' | Chunks: %d",
        disease_name, len(results),
    )

    return results
