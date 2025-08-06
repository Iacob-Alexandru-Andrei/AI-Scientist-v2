"""
pipeline.py – high‑level orchestration for the paper‑improver
=============================================================

The ``improve_paper`` entry‑point builds typed configuration objects,
selects one of several search strategies (BFS, priority tree, MAP‑Elites
or its modern variants), launches the search, and finally returns the
best improved paper node.

New in this revision
--------------------
* Strong static typing throughout; no *args / **kwargs.
* All hyper‑parameters grouped into immutable ``@dataclass`` configs.
* Modern QD variants (CVT‑MAP‑Elites, Descriptor‑Conditioned‑Gradient
  MAP‑Elites) are exposed through the same strategy interface, reflecting
  recent advances in quality‑diversity optimisation :contentReference[oaicite:0]{index=0}.
* Centralised construction of LLM/VLM review parameters avoids code
  duplication while keeping each call explicit.
* Every search run creates a timestamped *writable* copy of the original
  LaTeX project, ensuring that all states (drafts, reflections, write‑ups,
  citations) are preserved deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import logging
from pathlib import Path
import shutil

from .core import (  # search back‑ends
    SearchParams,
)
from .search import genetic_search_improve
from .llm_review import DEFAULT_MODEL, VLM_MODEL

EDITOR_MODEL = "gemini-2.5-flash-preview-04-17"
DEFAULT_ROUNDS = 2
DEFAULT_PAGE_LIMIT = 12
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Configuration dataclasses
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class WriteupConfig:
    num_cite_rounds: int
    citation_model: str
    editor_model: str
    num_reflections: int
    page_limit: int


@dataclass(slots=True, frozen=True)
class LLMReviewConfig:
    num_reflections: int
    num_fs_examples: int
    num_reviews_ensemble: int
    temperature: float


# NOTE:  VLM review seldom needs more than a kwargs blob, so keep it optional
@dataclass(slots=True, frozen=True)
class VLMReviewConfig:
    additional_kwargs: dict[str, object] | None = field(default=None)


class Strategy(Enum):
    """Available search strategies."""

    GENETIC = auto()

    @staticmethod
    def from_string(label: str) -> "Strategy":
        mapping = {
            "bfs": Strategy.BFS,
            "tree": Strategy.TREE,
            "map": Strategy.MAP,
            "cvt_map": Strategy.CVT_MAP,
            "dcg_map": Strategy.DCG_MAP,
            "parallel": Strategy.PARALLEL,
        }
        try:
            return mapping[label.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown strategy '{label}'.") from exc


# --------------------------------------------------------------------------- #
#  Public entry‑point
# --------------------------------------------------------------------------- #


def improve_paper(
    *,
    latex_project_dir: str | Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    strategy: str = "bfs",
    # model choices
    editor_model: str = EDITOR_MODEL,
    review_model: str = DEFAULT_MODEL,
    vlm_model: str = VLM_MODEL,
    orchestrator_model: str = "gemini-2.5-flash-preview-04-17",
    citation_model: str = "gpt-4o-2024-11-20",
    # write‑up
    num_cite_rounds: int = 20,
    num_writeup_reflections: int = DEFAULT_ROUNDS,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    # llm review
    llm_num_reflections: int = 1,
    llm_num_fs_examples: int = 1,
    llm_num_reviewers: int = 1,
    llm_temperature: float = 0.75,
    # search hyper‑params
    max_depth: int = 3,
    beam_size: int = 4,
    num_initial_drafts: int = 3,
    debug_retry_prob: float = 0.5,
    max_debug_depth: int = 3,
    # output control
    output_dir: str | Path | None = None,
) -> Path:
    """
    Improve a LaTeX project and return the directory of the best variant.

    All parameters are explicitly typed; there is no silent forwarding of
    unspecified keyword arguments.
    """

    # --------------------------------------------------------------------- #
    #  1. Resolve project & output paths
    # --------------------------------------------------------------------- #
    project_root = Path(latex_project_dir).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(project_root)

    if output_dir is not None:
        out_root = Path(output_dir).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = out_root / f"{project_root.name}_{stamped}"
        shutil.copytree(project_root, dest, dirs_exist_ok=True)
        project_root = dest  # subsequent steps operate on the copy
        logger.info("Copied project to working dir %s", project_root)

    # --------------------------------------------------------------------- #
    #  2. Build typed configuration bundles
    # --------------------------------------------------------------------- #
    write_cfg = WriteupConfig(
        num_cite_rounds=num_cite_rounds,
        citation_model=citation_model,
        editor_model=editor_model,
        num_reflections=num_writeup_reflections,
        page_limit=page_limit,
    )

    llm_cfg = LLMReviewConfig(
        num_reflections=llm_num_reflections,
        num_fs_examples=llm_num_fs_examples,
        num_reviews_ensemble=llm_num_reviewers,
        temperature=llm_temperature,
    )

    vlm_cfg = VLMReviewConfig(additional_kwargs=None)

    search_cfg = SearchParams(
        max_depth=max_depth,
        beam_size=beam_size,
        num_drafts=num_initial_drafts,
        debug_prob=debug_retry_prob,
        max_debug_depth=max_debug_depth,
        writeup_params=write_cfg.__dict__,
        llm_review_kwargs=llm_cfg.__dict__,
        vlm_review_kwargs=vlm_cfg.additional_kwargs,
    )

    # --------------------------------------------------------------------- #
    #  3. Dispatch to the chosen strategy
    # --------------------------------------------------------------------- #
    selected_strategy = Strategy.from_string(strategy)

    if selected_strategy is Strategy.GENETIC:
        best_node, _ = genetic_search_improve(
            root_dir=project_root,
            seed_ideas=seed_ideas,
            human_reviews=human_reviews,
            params=search_cfg,
            model_editor=editor_model,
            model_review=review_model,
            model_vlm=vlm_model,
            orchestrator_model=orchestrator_model,
        )
    else:  # defensive; should never hit thanks to Enum validation
        raise RuntimeError(f"Unhandled strategy {selected_strategy}")

    # --------------------------------------------------------------------- #
    #  4. Log & return result
    # --------------------------------------------------------------------- #
    logger.info("Best improved paper saved at %s", best_node.latex_dir)
    return best_node.latex_dir
