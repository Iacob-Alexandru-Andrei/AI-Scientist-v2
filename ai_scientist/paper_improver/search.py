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
    """Hyper-parameters controlling tree search behavior.

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
    description="Select the best implementation based on comprehensive analysis",
    json_schema={
        "type": "object",
        "properties": {
            "selected_id": {
                "type": "string",
                "description": "ID of the selected best implementation",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed explanation of why this implementation was chosen",
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
                f"ID: {n.id} Score: {n.score:.3f} paper text: {n.tex_path.read_text()} paper reviews: {str(n.llm_jsons) + str(n.vlm_json)} \n"
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


def breadth_first_improve(
    root_dir: Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    *,
    params: SearchParams | None = None,
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
    writeup_params: dict | None = None,
    llm_review_kwargs: dict | None = None,
    vlm_review_kwargs: dict | None = None,
):
    """Explore paper edits using a breadth-first expansion order."""
    p = params or SearchParams(max_depth=3, beam_size=4)
    if writeup_params is None:
        writeup_params = p.writeup_params
    if llm_review_kwargs is None:
        llm_review_kwargs = p.llm_review_kwargs or {}
    if vlm_review_kwargs is None:
        vlm_review_kwargs = p.vlm_review_kwargs or {}
    # The root node corresponds to the initial paper.  It is evaluated once so
    # the search has a baseline score.
    root = PaperNode(root_dir, llm_model=model_review, vlm_model=model_vlm)
    safe_evaluate(root, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
    journal = Journal()  # track every explored node for later selection
    journal.append(root)
    frontier = deque([root])
    best_state = root
    # Create a number of draft children before entering the main loop
    for i in range(p.num_drafts):
        draft_dir = unique_subdir(root.latex_dir.parent.parent, "draft")
        shutil.copytree(root.latex_dir, draft_dir)
        tex_path = draft_dir / "template.tex"
        new_source = propose_edit(
            tex_path,
            seed_ideas,
            model_reviews=str(root.llm_jsons) + str(root.vlm_json),
            human_reviews=human_reviews,
            model=model_editor,
        )  # model proposes a complete LaTeX rewrite of template.tex
        tex_path.write_text(new_source)
        if writeup_params is not None:
            tex_content = tex_path.read_text()
            citations_text = gather_citations(
                draft_dir,
                num_cite_rounds=writeup_params["num_cite_rounds"],
                small_model=writeup_params["small_model"],
            )
            if citations_text:
                pattern_end = r"\end{filecontents}"
                tex_content = tex_content.replace(
                    pattern_end,
                    f"\n{citations_text}{pattern_end}",
                )
                tex_path.write_text(tex_content)

        draft = PaperNode(
            draft_dir,
            root.depth + 1,
            parent=root,
            llm_model=model_review,
            vlm_model=model_vlm,
        )
        root.children.append(draft)  # keep a tree structure for analysis
        safe_evaluate(draft, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
        journal.append(draft)
        frontier.append(draft)
    # Standard BFS loop
    # Main priority queue loop
    while frontier:
        state = frontier.popleft()
        logger.info("Evaluating depth=%s dir=%s", state.depth, state.latex_dir)
        score = (
            state.score
            if state is root
            else safe_evaluate(
                state, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs
            )
        )
        if score and best_state.score and score > best_state.score:
            best_state = state
            logger.info("[NEW BEST] score=%.3f at %s", score, state.latex_dir)
        if state.depth >= p.max_depth:
            continue
        # If evaluation failed we may ask the editor model to try again.
        if (
            state.is_buggy
            and state.debug_depth < p.max_debug_depth
            and random.random() < p.debug_prob
        ):
            tex_path = state.latex_dir / "template.tex"
            new_source = propose_edit(
                tex_path,
                seed_ideas,
                model_reviews=str(root.llm_jsons) + str(root.vlm_json),
                human_reviews=human_reviews,
                model=model_editor,
            )  # attempt to fix the buggy document
            tex_path.write_text(new_source)
            state.debug_depth += 1
            if writeup_params is not None:
                perform_writeup(state.latex_dir, **writeup_params)
            safe_evaluate(
                state, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs
            )
        # Generate ``beam_size`` children by applying LLM edits to the current
        # LaTeX source
        for i in range(p.beam_size):
            child_dir = unique_subdir(state.latex_dir.parent, f"d{state.depth}")
            # Copy the current directory so the child starts with the same source
            shutil.copytree(state.latex_dir, child_dir)
            tex_path = child_dir / "template.tex"
            new_source = propose_edit(
                tex_path,
                seed_ideas,
                model_reviews=str(root.llm_jsons) + str(root.vlm_json),
                human_reviews=human_reviews,
                model=model_editor,
            )  # LLM suggests an updated LaTeX file
            # Overwrite the source so subsequent evaluation compiles the new version
            tex_path.write_text(new_source)
            if writeup_params is not None:
                tex_content = tex_path.read_text()
                citations_text = gather_citations(
                    child_dir,
                    num_cite_rounds=writeup_params["num_cite_rounds"],
                    small_model=writeup_params["small_model"],
                )
                if citations_text:
                    pattern_end = r"\end{filecontents}"
                    tex_content = tex_content.replace(
                        pattern_end,
                        f"\n{citations_text}{pattern_end}",
                    )
                    tex_path.write_text(tex_content)

            child = PaperNode(
                child_dir,
                state.depth + 1,
                parent=state,
                llm_model=model_review,
                vlm_model=model_vlm,
            )
            state.children.append(child)
            # Score the newly created child immediately so it can be queued
            safe_evaluate(
                child, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs
            )
            journal.append(child)
            frontier.append(child)
    # Let the orchestrator or fallback logic pick the best candidate overall
    return journal.best_node(orchestrator_model), journal


def tree_search_improve(
    root_dir: Path,
    human_reviews: str | None = None,
    *,
    params: SearchParams | None = None,
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
    num_cite_rounds: int = 2,
    llm_review_kwargs: dict | None = None,
    vlm_review_kwargs: dict | None = None,
    n_writeup_reflections=3,
    page_limit: int = 8,
):
    """Priority-based tree search over paper versions."""
    p = params or SearchParams(max_depth=3, beam_size=4)
    if llm_review_kwargs is None:
        llm_review_kwargs = p.llm_review_kwargs or {}
    if vlm_review_kwargs is None:
        vlm_review_kwargs = p.vlm_review_kwargs or {}
    root = PaperNode(root_dir / "latex", llm_model=model_review, vlm_model=model_vlm)
    safe_evaluate(root, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
    journal = Journal()
    journal.append(root)
    frontier: list[tuple[float, PaperNode]] = []
    # Generate a few initial drafts and push them onto the priority queue
    for i in range(p.num_drafts):
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
        frontier.append((draft.score, draft))  # min-heap by negative score

    # Main priority queue loop
    while frontier:
        frontier.sort()
        print([n[0] for n in frontier])
        _, node = frontier.pop()
        logger.info(
            "Exploring depth=%s dir=%s score=%.3f, latex_dir=%s, max depth=%s",
            node.depth,
            node.latex_dir,
            node.score,
            p.max_depth,
        )
        if node.depth >= p.max_depth:
            continue
        if (
            node.is_buggy
            and node.debug_depth < p.max_debug_depth
            and random.random() < p.debug_prob
        ):
            perform_writeup(
                base_folder=node.latex_dir.parent,
                model_reviews=str(root.llm_jsons) + str(root.vlm_json),
                human_reviews=human_reviews,
                num_cite_rounds=num_cite_rounds,
                small_model=model_vlm,
                big_model=model_editor,
                n_writeup_reflections=n_writeup_reflections,
                page_limit=page_limit,
            )
            node.debug_depth += 1
            safe_evaluate(
                node, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs
            )
        # Expand by proposing ``beam_size`` edits from the current node
        for i in range(p.beam_size):
            child_dir = unique_subdir(node.latex_dir.parent.parent, f"d{node.depth}")
            shutil.copytree(node.latex_dir.parent, child_dir)
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
            while not success:
                logger.warning(
                    "Writeup failed for %s, retrying with a new draft", child_dir
                )
                child_dir = unique_subdir(
                    node.latex_dir.parent.parent, f"d{node.depth}"
                )
                shutil.copytree(node.latex_dir.parent, child_dir)
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
                child_dir / "latex",
                node.depth + 1,
                parent=node,
                llm_model=model_review,
                vlm_model=model_vlm,
            )
            node.children.append(child)
            safe_evaluate(
                child, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs
            )
            journal.append(child)
            frontier.append((child.score, child))

    return journal.best_node(orchestrator_model), journal
