"""Generate and apply edit proposals to LaTeX source using an LLM."""
from pathlib import Path
from ai_scientist.llm import create_client
from ai_scientist.utils.token_tracker import track_token_usage

EDITOR_MODEL = "o1-preview-2024-09-12"


@track_token_usage
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
    prompt = f"""You are an expert academic writing assistant.  Below is the current LaTeX paper, a set of human reviews, and high-level improvement ideas.
Improve the document **in place** focusing on clarity, scientific rigour, and addressing reviewers’ concerns.  Output *only* the updated LaTeX in a fenced ```latex block.

############  CURRENT LaTeX ############
{latex_path.read_text()}
########################################

############ HUMAN REVIEWS ############
{human_reviews or 'N/A'}
########################################

############  SEED IDEAS   ############
{seed_ideas}
########################################
"""
    client, m = create_client(model)
    resp = client.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    # Extract LaTeX from fenced block – simple heuristic
    import re, textwrap
    code = re.search(r"```latex\s*(.*?)```", resp.choices[0].message.content, re.DOTALL)
    if not code:
        raise ValueError("No LaTeX block returned by editor model")
    new_source = textwrap.dedent(code.group(1)).strip()
    return new_source
