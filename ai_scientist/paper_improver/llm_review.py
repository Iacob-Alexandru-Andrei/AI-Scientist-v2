"""Wrapper around :mod:`perform_llm_review` for paper improvement.

The real heavy lifting happens in :func:`perform_llm_review.perform_review`.
This module simply exposes a thin convenience function ``llm_review`` that
forwards the desired model name and any keyword arguments.  ``create_client``
is used so the same code works for OpenAI, Gemini or Anthropic models.
"""

from typing import Any
from ai_scientist.perform_llm_review import perform_review  # existing util
from ai_scientist.llm import create_client

DEFAULT_MODEL = "gpt-4o-2024-11-20"


def llm_review(
    tex_or_pdf_path: str,
    *,
    model: str = DEFAULT_MODEL,
    **kwargs,
) -> dict[str, Any]:
    """Run the standard LLM review and return the parsed JSON.

    Parameters
    ----------
    tex_or_pdf_path
        Path to the LaTeX source (compiled) *or* an already-built PDF.
    model
        LLM to use for the review phase.
    """
    client, m = create_client(model)
    # Delegate the actual reviewing logic to ``perform_review``.  We simply pass
    # through the client and model name so downstream code remains decoupled
    # from any particular LLM provider.
    review_json = perform_review(
        tex_or_pdf_path, m, client, **kwargs, num_fs_examples=0
    )
    return review_json
