"""Search strategies for iteratively improving a LaTeX paper.

This module implements two algorithms:

``breadth_first_improve`` – a simple breadth-first traversal that expands all
nodes level by level.

``tree_search_improve`` – a priority based search that more closely resembles
the tree search used in the main AI-Scientist pipeline.

Both operate on ``PaperNode`` objects which hold paths to on-disk LaTeX
projects.  Each node is scored via LLM and VLM reviews and the resulting number
is used to rank candidates.  The ``Journal`` class records every explored node
and can optionally defer the final selection to an orchestration model.
"""

from __future__ import annotations

from collections import deque
import heapq
import uuid
import random
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import json
import logging
from ai_scientist import llm
from ai_scientist.perform_icbinb_writeup import gather_citations
from ai_scientist.treesearch.backend import query, FunctionSpec
from .latex_editor import propose_edit, EDITOR_MODEL
from .llm_review import llm_review, DEFAULT_MODEL
from .vlm_review import vlm_review, VLM_MODEL
from .meta_review import meta_score
from .utils import unique_subdir
from .writeup import perform_writeup

logger = logging.getLogger(__name__)


@dataclass
class SearchParams:
    """Hyper-parameters controlling genetic search behavior.

    ``max_depth``
        Maximum depth to explore in the search tree.
    ``beam_size``
        Number of children to generate from each node.
    ``num_drafts``
        How many initial drafts are spawned before the main loop.
    ``debug_prob`` and ``max_debug_depth``
        Control the probabilistic retry mechanism when a node fails to evaluate
        (e.g. LaTeX compilation errors).
    """

    max_depth: int = 3
    beam_size: int = 4
    num_drafts: int = 3
    debug_prob: float = 0.5
    max_debug_depth: int = 3
    # parameters for writeup generation after each edit
    writeup_params: dict | None = None
    # parameters forwarded to llm_review and vlm_review
    llm_review_kwargs: dict | None = None
    vlm_review_kwargs: dict | None = None


def safe_evaluate(
    node: "PaperNode",
    llm_kwargs: dict | None = None,
    vlm_kwargs: dict | None = None,
) -> float | None:
    """Evaluate a node and mark it buggy on failure."""
    try:
        return node.evaluate(llm_kwargs=llm_kwargs, vlm_kwargs=vlm_kwargs)
    except TypeError:
        # compatibility with older tests that mock evaluate without kwargs
        try:
            return node.evaluate()
        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", node.latex_dir, exc)
            node.is_buggy = True
            return None
    except Exception as exc:
        # Any exception during ``evaluate`` marks the node as buggy so the
        # search algorithms can attempt a debug retry.
        logger.error("Evaluation failed for %s: %s", node.latex_dir, exc)
        node.is_buggy = True
        return None


# Default model used to pick the best node once search has finished.  It takes
# a list of candidates and returns the selected node ID with some reasoning.
ORCHESTRATOR_MODEL = "gemini-2.5-flash-preview-04-17"

# Function specification describing the JSON schema for orchestrator output.
node_selection_spec = FunctionSpec(
    name="select_best_implementation",
    description="Select the best paper based on comprehensive analysis",
    json_schema={
        "type": "object",
        "properties": {
            "selected_id": {
                "type": "string",
                "description": "ID of the selected best paper",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed explanation of why this paper was chosen",
            },
        },
        "required": ["selected_id", "reasoning"],
    },
)


@dataclass(eq=False)
class PaperNode:
    """A paper version on disk (latex_dir contains template.tex)."""

    latex_dir: Path
    depth: int = 0
    parent: "PaperNode | None" = None
    llm_model: str = DEFAULT_MODEL
    vlm_model: str = VLM_MODEL
    debug_depth: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    score: float | None = None
    llm_jsons: tuple[dict, ...] = ()
    vlm_json: dict | None = None
    orchestrator_recommendations: str | None = None
    is_buggy: bool = False
    is_buggy_plots: bool = False
    step: int = 0
    children: list["PaperNode"] = field(default_factory=list)

    def __post_init__(self):
        self.pdf_path = self.latex_dir / "template.pdf"
        self.tex_path = self.latex_dir / "template.tex"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "latex_dir": str(self.latex_dir),
            "depth": self.depth,
            "parent_id": self.parent.id if self.parent else None,
            "llm_model": self.llm_model,
            "vlm_model": self.vlm_model,
            "score": self.score,
            "llm_jsons": self.llm_jsons,
            "vlm_json": self.vlm_json,
            "is_buggy": self.is_buggy,
            "is_buggy_plots": self.is_buggy_plots,
            "debug_depth": self.debug_depth,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: dict, journal: "Journal" | None = None) -> "PaperNode":
        parent = None
        if journal and data.get("parent_id"):
            parent = journal.get_node_by_id(data["parent_id"])
        node = cls(
            latex_dir=Path(data["latex_dir"]),
            depth=data.get("depth", 0),
            parent=parent,
            llm_model=data.get("llm_model", DEFAULT_MODEL),
            vlm_model=data.get("vlm_model", VLM_MODEL),
            debug_depth=data.get("debug_depth", 0),
        )
        node.id = data.get("id", node.id)
        node.score = data.get("score")
        node.llm_jsons = data.get("llm_jsons")
        node.vlm_json = data.get("vlm_json")
        node.is_buggy = data.get("is_buggy", False)
        node.is_buggy_plots = data.get("is_buggy_plots", False)
        node.step = data.get("step", 0)
        return node

    def compile(self):
        # ``compile_latex`` is reused from the main code base. It simply
        # invokes ``latexmk`` to build a PDF in ``self.pdf_path``.
        from ai_scientist.perform_icbinb_writeup import compile_latex  # reuse util

        compile_latex(str(self.latex_dir), str(self.pdf_path))

    def evaluate(
        self,
        *,
        llm_kwargs: dict | None = None,
        vlm_kwargs: dict | None = None,
    ):
        if not self.pdf_path.exists():
            self.compile()
        llm_kwargs = llm_kwargs or {}
        vlm_kwargs = vlm_kwargs or {}
        print(llm_kwargs)
        # Run LLM and VLM reviews then compute an aggregate numeric score.
        self.llm_jsons, _ = llm_review(
            self.tex_path.read_text(), model=self.llm_model, **llm_kwargs
        )
        self.vlm_json = vlm_review(
            str(self.pdf_path), model=self.vlm_model, **vlm_kwargs
        )
        self.score = meta_score(self.llm_jsons + (self.vlm_json,))
        # Persist results for analysis
        with open(self.latex_dir.parent / "reviews.json", "w") as f:
            llm_json_dump = {}
            for i, llm_json in enumerate(self.llm_jsons):
                llm_json_dump["id"] = f"llm_{i}"
                llm_json_dump["review"] = llm_json
            json.dump(
                {"llm": llm_json_dump, "vlm": self.vlm_json, "score": self.score},
                f,
                indent=2,
            )
        return self.score


class Journal:
    """Keep track of explored paper versions."""

    def __init__(self) -> None:
        self.nodes: list[PaperNode] = []

    def append(self, node: PaperNode) -> None:
        # ``step`` mirrors the attribute used by the original tree search
        # implementation and simply records insertion order.
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def good_nodes(self) -> list[PaperNode]:
        return [n for n in self.nodes if not n.is_buggy and not n.is_buggy_plots]

    @property
    def buggy_nodes(self) -> list[PaperNode]:
        return [n for n in self.nodes if n.is_buggy]

    def get_node_by_id(self, node_id: str) -> PaperNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    @property
    def draft_nodes(self) -> list[PaperNode]:
        """Nodes without parents (initial drafts)."""
        return [n for n in self.nodes if n.parent is None]

    # Backwards compatibility with the original tree-search API
    def get_best_node(
        self, orchestrator_model: str = ORCHESTRATOR_MODEL
    ) -> PaperNode | None:
        return self.best_node(orchestrator_model)

    def best_node(
        self, orchestrator_model: str = ORCHESTRATOR_MODEL
    ) -> PaperNode | None:
        if not self.nodes:
            return None
        if len(self.nodes) == 1:
            return self.nodes[0]

        # Construct a small system prompt summarising each candidate's score.
        prompt = {
            "Introduction": (
                "You are an experienced researcher choosing the best improved paper version based on review scores, text and reviews"
            ),
            "Candidates": "",
        }
        for n in self.nodes:
            prompt["Candidates"] += (
                f"ID: {n.id} Score: {n.score:.3f} paper text: {n.tex_path.read_text()} paper reviews: {str(n.llm_json) + str(n.vlm_json)} \n"
            )

        try:
            selection = query(
                system_message=prompt,
                user_message=None,
                func_spec=node_selection_spec,
                model=orchestrator_model,
                temperature=0.3,
            )
            selected = next(
                (n for n in self.nodes if n.id == selection["selected_id"]), None
            )
            if selected:
                return selected
        except Exception as exc:
            # If the orchestrator call fails we simply fall back to a numeric
            # best-node selection so the search does not crash.
            logger.error("Orchestrator selection failed: %s", exc)
        return max(self.nodes, key=lambda n: n.score or 0)


def genetic_improve(
    root_dir: Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    *,
    params: SearchParams | None = None,
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
    tournament_size: int = 3,
    num_elites: int = 2,
    llm_review_kwargs: dict | None = None,
    vlm_review_kwargs: dict | None = None,
    num_cite_rounds: int = 2,
    n_writeup_reflections: int = 3,
    page_limit: int = 8,
    num_iter: int = 100,
) -> tuple[PaperNode | None, Journal]:
    """Search using a simple MAP-ELITES strategy."""

    p = params or SearchParams(max_depth=3, beam_size=4)
    if llm_review_kwargs is None:
        llm_review_kwargs = p.llm_review_kwargs or {}
    if vlm_review_kwargs is None:
        vlm_review_kwargs = p.vlm_review_kwargs or {}

    root = PaperNode(root_dir, llm_model=model_review, vlm_model=model_vlm)
    safe_evaluate(root, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
    journal = Journal()
    journal.append(root)

    elites: dict[int, PaperNode] = {}

    def add_elite(node: PaperNode) -> None:
        idx = int((node.score or 0) // 1)
        current = elites.get(idx)
        if current is None or (node.score or 0) > (current.score or -1):
            elites[idx] = node

    add_elite(root)

    # Generate a few initial drafts and push them onto the priority queue
    for _ in range(p.num_drafts):
        draft_dir = unique_subdir(root.latex_dir.parent.parent, "draft")
        shutil.copytree(root.latex_dir.parent, draft_dir)
        success = perform_writeup(
            base_folder=draft_dir,
            model_reviews=str(root.llm_jsons) + str(root.vlm_json),
            human_reviews=human_reviews,
            num_cite_rounds=num_cite_rounds,
            small_model=model_vlm,
            big_model=model_editor,
            n_writeup_reflections=n_writeup_reflections,
            page_limit=page_limit,
        )
        while not success:
            logger.warning(
                "Writeup failed for %s, retrying with a new draft", draft_dir
            )
            draft_dir = unique_subdir(root.latex_dir.parent.parent, "draft")
            shutil.copytree(root.latex_dir.parent, draft_dir)
            success = perform_writeup(
                base_folder=draft_dir,
                model_reviews=str(root.llm_jsons) + str(root.vlm_json),
                human_reviews=human_reviews,
                num_cite_rounds=num_cite_rounds,
                small_model=model_vlm,
                big_model=model_editor,
                n_writeup_reflections=n_writeup_reflections,
                page_limit=page_limit,
            )
        draft = PaperNode(
            draft_dir / "latex",
            root.depth + 1,
            parent=root,
            llm_model=model_review,
            vlm_model=model_vlm,
        )
        root.children.append(draft)
        safe_evaluate(draft, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
        journal.append(draft)
        add_elite(draft)

    depth = 0

    for _ in range(num_iter):
        # Tournament selection among elites
        candidates = random.sample(
            list(elites.values()), min(tournament_size, len(elites))
        )
        from ai_scientist.perform_llm_review import neurips_form

        prompt = {
            "Introduction": "Select the best paper according to the received reviews created using the following guidelines",
            "Review guidelines": neurips_form,
            "Candidates": "",
        }
        for n in candidates:
            prompt["Candidates"] += (
                f"ID: {n.id} Score: {n.score}, reviews: {n.llm_jsons + (n.vlm_json,)}, text: {n.tex_path.read_text()}\n"
            )
        try:
            choice = query(
                system_message=prompt,
                user_message=None,
                func_spec=node_selection_spec,
                model=orchestrator_model,
                temperature=0.3,
            )
            parent = next(
                (n for n in candidates if n.id == choice["selected_id"]), None
            )
            parents_b = ([n for n in candidates if n.id != choice["selected_id"]],)
            prompt = prompt = {
                "Orchestration guidelines": orchestrator_prompt,
                "Candidates": "",
            }

            recommended_improvements = llm.get_response_from_llm(
                orchestrator_prompt,
                model=model_editor,
                client=llm.create_client(model_editor),
                system_message="Be as harsh as possible, focus on the most important improvements that could get the paper to be accepted at a top conference or journal in Machine Learning.",
                print_debug=False,
                msg_history=None,
                temperature=0.75,
            )

        except Exception:
            parent_b = max(candidates, key=lambda n: n.score or 0)

        child_dir = unique_subdir(state.latex_dir.parent, f"m{depth}")
        shutil.copytree(state.latex_dir, child_dir)
        tex_path = child_dir / "template.tex"

        success = perform_writeup(
            base_folder=child_dir,
            model_reviews=str(root.llm_jsons) + str(root.vlm_json),
            human_reviews=human_reviews,
            num_cite_rounds=num_cite_rounds,
            small_model=model_vlm,
            big_model=model_editor,
            n_writeup_reflections=n_writeup_reflections,
            page_limit=page_limit,
        )

        child = PaperNode(
            child_dir,
            depth + 1,
            parent=state,
            llm_model=model_review,
            vlm_model=model_vlm,
        )
        state.children.append(child)
        safe_evaluate(child, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
        journal.append(child)
        add_elite(child)
        next_frontier.append(child)

    return journal.best_node(orchestrator_model), journal
