"""Thin wrapper around :func:`perform_imgs_cap_ref_review`.

The :mod:`perform_vlm_review` module contains the heavy logic for assessing a
paper's figures and captions with a vision-language model.  Here we expose a
single ``vlm_review`` function that selects the model and passes through the
PDF path.  This mirrors ``llm_review`` for textual reviews.
"""

from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.vlm import create_client as create_vlm_client

VLM_MODEL = "gpt-4o-2024-11-20"


def vlm_review(pdf_path: str, *, model: str = VLM_MODEL, **kwargs) -> dict:
    """Run the vision-language review over a compiled PDF."""
    client, m = create_vlm_client(model)
    return perform_imgs_cap_ref_review(client, m, pdf_path, **kwargs)
