"""LLM-only review wrapper."""
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
    review_json = perform_review(tex_or_pdf_path, m, client, **kwargs)
    return review_json
