"""High-level orchestration: given inputs, run the improver search."""

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
from ai_scientist.perform_icbinb_writeup import gather_citations

CITATION_MODEL = "gpt-4o-2024-11-20"
DEFAULT_CITE_ROUNDS = 20

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
    max_depth: int = 3,
    beam_size: int = 4,
    num_drafts: int = 3,
    debug_prob: float = 0.5,
    max_debug_depth: int = 3,
    **kwargs,
):
    root = Path(latex_project_dir).resolve()
    params = SearchParams(
        max_depth=max_depth,
        beam_size=beam_size,
        num_drafts=num_drafts,
        debug_prob=debug_prob,
        max_debug_depth=max_debug_depth,
    )
    if strategy == "tree":
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
    if num_cite_rounds > 0:
        logger.info(
            "Gathering citations with %s for %d rounds", model_citation, num_cite_rounds
        )
        gather_citations(
            best_state.latex_dir,
            num_cite_rounds=num_cite_rounds,
            small_model=model_citation,
        )
    logger.info("Best improved paper saved at %s", best_state.latex_dir)
    return best_state
