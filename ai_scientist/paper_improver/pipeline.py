"""High-level orchestration: given inputs, run the improver search."""
from pathlib import Path
from .search import breadth_first_improve, tree_search_improve, ORCHESTRATOR_MODEL
from .latex_editor import EDITOR_MODEL
from .llm_review import DEFAULT_MODEL
from .vlm_review import VLM_MODEL


def improve_paper(
    latex_project_dir: str | Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    strategy: str = "bfs",
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
    **kwargs,
):
    root = Path(latex_project_dir).resolve()
    if strategy == "tree":
        best_state, _journal = tree_search_improve(
            root,
            seed_ideas,
            human_reviews,
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
            model_editor=model_editor,
            model_review=model_review,
            model_vlm=model_vlm,
            orchestrator_model=orchestrator_model,
            **kwargs,
        )
    print("Best improved paper saved at", best_state.latex_dir)
    return best_state
