"""High level orchestration for the paper improver.

``improve_paper`` is the entry point used by the CLI and tests.  It builds a
``SearchParams`` instance from the supplied hyper-parameters, chooses the search
strategy (breadth first or priority tree search) and then performs optional
citation gathering and reflection on the best node.
"""

from pathlib import Path
import logging
from .search import (
    breadth_first_improve,
    tree_search_improve,
    ORCHESTRATOR_MODEL,
    SearchParams,
)
from .latex_editor import EDITOR_MODEL
from .llm_review import DEFAULT_MODEL
from .vlm_review import VLM_MODEL
from .reflection import (
    reflect_paper,
    REFLECTION_MODEL,
    DEFAULT_ROUNDS as DEFAULT_REFLECTION_ROUNDS,
    DEFAULT_PAGE_LIMIT,
)
from ai_scientist.perform_icbinb_writeup import gather_citations

CITATION_MODEL = "gpt-4o-2024-11-20"
DEFAULT_CITE_ROUNDS = 20
REFLECTION_MODEL_DEFAULT = REFLECTION_MODEL
DEFAULT_REFLECTION_ROUNDS = DEFAULT_REFLECTION_ROUNDS
DEFAULT_PAGE_LIMIT_VALUE = DEFAULT_PAGE_LIMIT

logger = logging.getLogger(__name__)


def improve_paper(
    latex_project_dir: str | Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    strategy: str = "bfs",
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
    model_citation: str = CITATION_MODEL,
    num_cite_rounds: int = DEFAULT_CITE_ROUNDS,
    model_reflection: str = REFLECTION_MODEL_DEFAULT,
    num_reflections: int = DEFAULT_REFLECTION_ROUNDS,
    page_limit: int = DEFAULT_PAGE_LIMIT_VALUE,
    max_depth: int = 3,
    beam_size: int = 4,
    num_drafts: int = 3,
    debug_prob: float = 0.5,
    max_debug_depth: int = 3,
    **kwargs,
):
    # Resolve paths early to avoid confusion when the working directory changes
    root = Path(latex_project_dir).resolve()

    # Bundle all search-related hyper parameters into a dataclass instance
    params = SearchParams(
        max_depth=max_depth,
        beam_size=beam_size,
        num_drafts=num_drafts,
        debug_prob=debug_prob,
        max_debug_depth=max_debug_depth,
    )
    if strategy == "tree":
        # Priority-based tree search closely mirrors the main AI Scientist
        # implementation but skips experiment execution.
        best_state, _journal = tree_search_improve(
            root,
            seed_ideas,
            human_reviews,
            params=params,
            model_editor=model_editor,
            model_review=model_review,
            model_vlm=model_vlm,
            orchestrator_model=orchestrator_model,
            **kwargs,
        )
    else:
        # The default strategy explores states in a breadth-first manner.
        best_state, _journal = breadth_first_improve(
            root,
            seed_ideas,
            human_reviews,
            params=params,
            model_editor=model_editor,
            model_review=model_review,
            model_vlm=model_vlm,
            orchestrator_model=orchestrator_model,
            **kwargs,
        )
    # Optionally refine citations for the best candidate
    if num_cite_rounds > 0:
        logger.info(
            "Gathering citations with %s for %d rounds", model_citation, num_cite_rounds
        )
        gather_citations(
            best_state.latex_dir,
            num_cite_rounds=num_cite_rounds,
            small_model=model_citation,
        )
    # Final reflection loop to polish LaTeX and check page limits
    if num_reflections > 0:
        logger.info(
            "Running %d reflection rounds with %s", num_reflections, model_reflection
        )
        reflect_paper(
            best_state.latex_dir,
            model=model_reflection,
            vlm_model=model_vlm,
            num_rounds=num_reflections,
            page_limit=page_limit,
        )
    # Return the path to the best version for further inspection
    logger.info(
        "Best improved paper saved at %s %s", best_state.latex_dir, best_state.pdf_path
    )
    return best_state
