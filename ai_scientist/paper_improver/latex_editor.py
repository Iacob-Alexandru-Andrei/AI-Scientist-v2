"""Utilities for editing LaTeX using a language model.

The ``propose_edit`` function is a very small wrapper around the standard
``create_client`` helper.  It sends the current LaTeX source, optional
human reviews and a list of seed ideas to an LLM, then parses the model's
response for a fenced `````latex`` block.  The extracted code is returned
without writing it to disk.  Callers are responsible for updating the
``template.tex`` file themselves.
"""

from pathlib import Path
from ai_scientist.llm import create_client
from ai_scientist.utils.token_tracker import track_token_usage

EDITOR_MODEL = "o1-preview-2024-09-12"


# @track_token_usage
def propose_edit(
    latex_path: Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    model: str = EDITOR_MODEL,
) -> str:
    """Return *new* LaTeX code after applying improvements suggested by the model.

    Parameters
    ----------
    latex_path
        Path to the LaTeX source to edit in place.
    seed_ideas
        High-level suggestions guiding the improvement.
    human_reviews
        Optional human reviewer comments to address.
    model
        Model used to propose edits.
    """
    # Build a single prompt containing the source, reviews and improvement ideas
    # so the model can propose an updated version of the document.
    prompt = f"""You are an expert academic writing assistant.  Below is the current LaTeX paper, a set of human reviews, and high-level improvement ideas.
Improve the document **in place** focusing on clarity, scientific rigour, and addressing reviewers’ concerns.  Output *only* the updated LaTeX in a fenced ```latex block.

############  CURRENT LaTeX ############
{latex_path.read_text()}
########################################

############ HUMAN REVIEWS ############
{human_reviews or "N/A"}
########################################

############  SEED IDEAS   ############
{seed_ideas}
########################################
"""
    # Send the prompt to the selected model and obtain a completion.
    client, m = create_client(model)
    resp = client.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    # Extract LaTeX from fenced block – simple heuristic
    import re, textwrap

    # The model is instructed to place the revised LaTeX in a fenced block.
    # A simple non-greedy regex is used to capture that code snippet.
    code = re.search(r"```latex\s*(.*?)```", resp.choices[0].message.content, re.DOTALL)
    if not code:
        raise ValueError("No LaTeX block returned by editor model")
    # Normalise indentation just in case the model added leading spaces.
    new_source = textwrap.dedent(code.group(1)).strip()
    return new_source
