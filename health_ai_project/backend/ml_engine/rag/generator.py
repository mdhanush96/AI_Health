"""
MedAI – FLAN-T5 Context-Grounded Generator
Takes retrieved medical chunks + user query and generates a structured
natural-language medical explanation using FLAN-T5.
"""

import logging
import time

import torch

from ml_engine import config
from .rag_loader import load_generator_model

logger = logging.getLogger("ml_engine.rag")


def _build_prompt(
    query: str,
    retrieved_chunks: list[dict],
    classified_disease: str | None = None,
) -> str:
    """
    Build a focused instruction-tuned prompt for FLAN-T5.
    Keeps prompt concise to avoid truncation — FLAN-T5-base works best
    with shorter, focused instructions.
    """
    # Collect unique disease-specific context (deduplicate by disease)
    seen_diseases = set()
    context_parts = []
    for chunk in retrieved_chunks:
        d = chunk["disease"]
        if d not in seen_diseases:
            seen_diseases.add(d)
            # Take first 800 chars per disease for richer context
            content = chunk["content"][:800]
            context_parts.append(f"{d}: {content}")
        if len(context_parts) >= 5:
            break
    context_block = "\n\n".join(context_parts)

    disease_line = ""
    if classified_disease:
        disease_line = f"Predicted condition: {classified_disease}.\n"

    prompt = (
        f"You are a medical expert. Based on the medical knowledge below, "
        f"write a comprehensive explanation about the patient's likely condition.\n\n"
        f"Patient symptoms: {query}\n"
        f"{disease_line}\n"
        f"Medical knowledge:\n{context_block}\n\n"
        f"Write a thorough medical explanation covering:\n"
        f"1. What the condition is and how it develops\n"
        f"2. The main causes and risk factors\n"
        f"3. Key symptoms to watch for\n"
        f"4. Possible complications if untreated\n"
        f"5. Recommended lifestyle changes and dietary modifications\n"
        f"6. When to seek emergency medical care\n\n"
        f"Detailed explanation:"
    )
    return prompt


def generate(
    query: str,
    retrieved_chunks: list[dict],
    classified_disease: str | None = None,
    max_output_tokens: int | None = None,
) -> dict:
    """
    Generate a context-grounded medical response using FLAN-T5.

    Args:
        query: User symptom text.
        retrieved_chunks: List from retriever.retrieve().
        classified_disease: Optional top prediction from ClinicalBERT.
        max_output_tokens: Override config default.

    Returns:
        dict with keys: generated_text, prompt_tokens, output_tokens, latency_ms
    """
    max_out = max_output_tokens or config.GENERATOR_MAX_OUTPUT_TOKENS

    model, tokenizer, device = load_generator_model()

    t0 = time.perf_counter()

    prompt = _build_prompt(query, retrieved_chunks, classified_disease)

    # Tokenize with truncation to fit model context window
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.GENERATOR_MAX_INPUT_TOKENS,
        padding=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_tokens = input_ids.shape[1]

    # Generate with no_grad — use sampling for richer output
    with torch.no_grad():
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_out,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        if config.GENERATOR_DO_SAMPLE:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": config.GENERATOR_TEMPERATURE,
                "top_p": config.GENERATOR_TOP_P,
                "top_k": 50,
            })
        else:
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": config.GENERATOR_NUM_BEAMS,
                "length_penalty": 1.2,
            })
        output_ids = model.generate(**gen_kwargs)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_tokens = output_ids.shape[1]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Generation complete | Prompt tokens: %d | Output tokens: %d | %.1fms",
        prompt_tokens, output_tokens, elapsed_ms,
    )

    return {
        "generated_text": generated_text.strip(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "latency_ms": round(elapsed_ms, 1),
    }
