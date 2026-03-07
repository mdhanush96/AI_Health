"""
MedAI – Integration Test Script
Tests the full Hybrid RAG pipeline end-to-end.
"""

import os
import sys
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "health_ai.settings")

import django
django.setup()


def test_classification():
    print("=" * 60)
    print("TEST 1: ClinicalBERT Classification")
    print("=" * 60)
    from ml_engine.predictor import predict

    t0 = time.perf_counter()
    result = predict("I have fever, headache and body pain")
    elapsed = (time.perf_counter() - t0) * 1000

    for p in result["predictions"]:
        print(f"  {p['disease']}: {p['confidence']}% ({p['risk_level']})")
    print(f"  Latency: {elapsed:.1f}ms")
    print("  PASSED\n")
    return result


def test_retriever():
    print("=" * 60)
    print("TEST 2: FAISS Semantic Retriever")
    print("=" * 60)
    from ml_engine.rag.retriever import retrieve

    t0 = time.perf_counter()
    chunks = retrieve("fever headache body pain", top_k=3)
    elapsed = (time.perf_counter() - t0) * 1000

    for c in chunks:
        print(f"  [{c['rank']}] {c['disease']} (score={c['score']}) — {c['content'][:80]}...")
    print(f"  Latency: {elapsed:.1f}ms")
    print("  PASSED\n")
    return chunks


def test_generator(chunks, disease):
    print("=" * 60)
    print("TEST 3: FLAN-T5 Generator")
    print("=" * 60)
    from ml_engine.rag.generator import generate

    gen = generate(
        "I have fever, headache and body pain",
        chunks,
        classified_disease=disease,
    )

    print(f"  Generated ({gen['output_tokens']} tokens, {gen['latency_ms']}ms):")
    print(f"  {gen['generated_text'][:300]}")
    print("  PASSED\n")
    return gen


def test_full_pipeline():
    print("=" * 60)
    print("TEST 4: Full Hybrid RAG Pipeline")
    print("=" * 60)
    from ml_engine.predictor import predict_with_rag

    result = predict_with_rag("I have fever, headache and body pain")
    print(f"  Top disease: {result['predictions'][0]['disease']}")
    print(f"  RAG response: {result['rag']['rag_response'][:300]}...")
    print(f"  Retrieved chunks: {len(result['rag']['retrieved_chunks'])}")
    print(f"  Cache hit: {result['rag']['cache_hit']}")
    print(f"  Total latency: {result['total_latency_ms']}ms")
    print("  PASSED\n")
    return result


def test_rag_status():
    print("=" * 60)
    print("TEST 5: RAG Component Status")
    print("=" * 60)
    from ml_engine.rag.rag_loader import get_rag_status

    status = get_rag_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\nMedAI Hybrid RAG – Integration Test Suite\n")

    clf_result = test_classification()
    chunks = test_retriever()
    test_generator(chunks, clf_result["predictions"][0]["disease"])
    test_full_pipeline()
    test_rag_status()

    print("=" * 60)
    print("ALL TESTS PASSED — System is operational.")
    print("=" * 60)
