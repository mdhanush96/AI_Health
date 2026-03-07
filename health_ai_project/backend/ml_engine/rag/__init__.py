"""
MedAI – RAG Module
Hybrid Retrieval-Augmented Generation for medical knowledge.

Submodules:
    rag_loader  — Singleton loaders for embedding model, FAISS index, FLAN-T5
    retriever   — Semantic FAISS retrieval
    generator   — FLAN-T5 context-grounded generation
    pipeline    — Orchestrator combining retriever + generator
"""

from .pipeline import rag_query  # noqa: F401
