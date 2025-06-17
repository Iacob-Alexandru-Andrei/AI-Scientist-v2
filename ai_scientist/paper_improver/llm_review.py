"""Wrapper around :mod:`perform_llm_review` for paper improvement.

The real heavy lifting happens in :func:`perform_llm_review.perform_review`.
This module simply exposes a thin convenience function ``llm_review`` that
forwards the desired model name and any keyword arguments.  ``create_client``
is used so the same code works for OpenAI, Gemini or Anthropic models.
"""

from typing import Any
from ai_scientist.perform_llm_review import perform_review
from ai_scientist.llm import create_client

DEFAULT_MODEL = "gpt-4o-2024-11-20"


def llm_review(
    tex_or_pdf_path: str,
    *,
    model: str = DEFAULT_MODEL,
    num_reflections: int = 1,
    num_fs_examples: int = 1,
    num_reviews_ensemble: int = 1,
    temperature: float = 0.75,
    msg_history: list | None = None,
    return_msg_history: bool = False,
    reviewer_system_prompt=None,
    review_instruction_form=None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Run the standard LLM review and return the parsed JSON.

    Parameters
    ----------
    tex_or_pdf_path
        Path to the LaTeX source (compiled) *or* an already-built PDF.
    model
        LLM to use for the review phase.
    """
    try:
        from ai_scientist.perform_llm_review import (
            reviewer_system_prompt_neg,
            neurips_form,
        )
    except Exception:  # pragma: no cover - fallback when deps missing
        reviewer_system_prompt_neg = "You are an AI reviewer."
        neurips_form = "Review:"

    if reviewer_system_prompt is None:
        reviewer_system_prompt = reviewer_system_prompt_neg
    if review_instruction_form is None:
        review_instruction_form = neurips_form

    client, m = create_client(model)
    # Delegate the actual reviewing logic to ``perform_review``.  We simply pass
    # through the client and model name so downstream code remains decoupled
    # from any particular LLM provider.
    review_json = perform_review(
        tex_or_pdf_path,
        m,
        client,
        num_reflections=num_reflections,
        num_fs_examples=num_fs_examples,
        num_reviews_ensemble=num_reviews_ensemble,
        temperature=temperature,
        msg_history=msg_history,
        return_msg_history=return_msg_history,
        reviewer_system_prompt=reviewer_system_prompt,
        review_instruction_form=review_instruction_form,
    )
    return review_json
