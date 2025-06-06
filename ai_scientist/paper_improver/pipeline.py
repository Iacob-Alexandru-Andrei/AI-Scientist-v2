"""High-level orchestration: given inputs, run the improver search."""
from pathlib import Path
from .search import breadth_first_improve, tree_search_improve


def improve_paper(
    latex_project_dir: str | Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    strategy: str = "bfs",
    **kwargs,
):
    root = Path(latex_project_dir).resolve()
    if strategy == "tree":
        best_state, _journal = tree_search_improve(
            root, seed_ideas, human_reviews, **kwargs
        )
    else:
        best_state, _journal = breadth_first_improve(
            root, seed_ideas, human_reviews, **kwargs
        )
    print("Best improved paper saved at", best_state.latex_dir)
    return best_state
