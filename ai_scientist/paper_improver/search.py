"""Breadth-First Tree Search (BFTS) over paper versions."""

from __future__ import annotations

from collections import deque
import heapq
import uuid
import random
from dataclasses import dataclass
from pathlib import Path
import shutil
import json
import logging
from ai_scientist.treesearch.backend import query, FunctionSpec
from .latex_editor import propose_edit, EDITOR_MODEL
from .llm_review import llm_review, DEFAULT_MODEL
from .vlm_review import vlm_review, VLM_MODEL
from .meta_review import meta_score
from .utils import unique_subdir

logger = logging.getLogger(__name__)


@dataclass
class SearchParams:
    """Hyper-parameters controlling tree search behavior."""

    max_depth: int = 3
    beam_size: int = 4
    num_drafts: int = 3
    debug_prob: float = 0.5
    max_debug_depth: int = 3


def safe_evaluate(node: "PaperNode") -> float | None:
    """Evaluate a node and mark it buggy on failure."""
    try:
        return node.evaluate()
    except Exception as exc:
        logger.error("Evaluation failed for %s: %s", node.latex_dir, exc)
        node.is_buggy = True
        return None


ORCHESTRATOR_MODEL = "gpt-4o-2024-11-20"

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


class PaperNode:
    """A paper version on disk (latex_dir contains template.tex)."""

    def __init__(
        self,
        latex_dir: Path,
        depth: int = 0,
        parent: "PaperNode | None" = None,
        llm_model: str = DEFAULT_MODEL,
        vlm_model: str = VLM_MODEL,
        debug_depth: int = 0,
    ):
        self.id = uuid.uuid4().hex
        self.latex_dir = latex_dir
        self.depth = depth
        self.parent = parent
        self.llm_model = llm_model
        self.vlm_model = vlm_model
        self.children: list["PaperNode"] = []
        self.pdf_path = latex_dir / "template.pdf"  # compiled later
        self.score: float | None = None
        self.llm_json: dict | None = None
        self.vlm_json: dict | None = None
        # compatibility with treesearch Journal
        self.is_buggy = False
        self.is_buggy_plots = False
        self.debug_depth = debug_depth

    def compile(self):
        from ai_scientist.perform_icbinb_writeup import compile_latex  # reuse util

        compile_latex(str(self.latex_dir), str(self.pdf_path))

    def evaluate(self):
        if not self.pdf_path.exists():
            self.compile()
        self.llm_json = llm_review(str(self.pdf_path), model=self.llm_model)
        self.vlm_json = vlm_review(str(self.pdf_path), model=self.vlm_model)
        self.score = meta_score([self.llm_json, self.vlm_json])
        # Persist results for analysis
        with open(self.latex_dir / "reviews.json", "w") as f:
            json.dump(
                {"llm": self.llm_json, "vlm": self.vlm_json, "score": self.score},
                f,
                indent=2,
            )
        return self.score


class Journal:
    """Keep track of explored paper versions."""

    def __init__(self) -> None:
        self.nodes: list[PaperNode] = []

    def append(self, node: PaperNode) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def best_node(
        self, orchestrator_model: str = ORCHESTRATOR_MODEL
    ) -> PaperNode | None:
        if not self.nodes:
            return None
        if len(self.nodes) == 1:
            return self.nodes[0]

        prompt = {
            "Introduction": (
                "You are an experienced researcher choosing the best improved paper version based on review scores."
            ),
            "Candidates": "",
        }
        for n in self.nodes:
            prompt["Candidates"] += f"ID: {n.id} Score: {n.score:.3f}\n"

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
):
    """Explore paper edits using best-first strategy."""
    p = params or SearchParams(max_depth=3, beam_size=4)
    root = PaperNode(root_dir, llm_model=model_review, vlm_model=model_vlm)
    safe_evaluate(root)
    journal = Journal()
    journal.append(root)
    frontier = deque([root])
    best_state = root
    # initial drafts
    for i in range(p.num_drafts):
        draft_dir = unique_subdir(root.latex_dir.parent, "draft")
        shutil.copytree(root.latex_dir, draft_dir)
        tex_path = draft_dir / "template.tex"
        new_source = propose_edit(
            tex_path, seed_ideas, human_reviews, model=model_editor
        )
        tex_path.write_text(new_source)
        draft = PaperNode(
            draft_dir,
            root.depth + 1,
            parent=root,
            llm_model=model_review,
            vlm_model=model_vlm,
        )
        root.children.append(draft)
        safe_evaluate(draft)
        journal.append(draft)
        frontier.append(draft)
    while frontier:
        state = frontier.popleft()
        logger.info("Evaluating depth=%s dir=%s", state.depth, state.latex_dir)
        score = state.score if state is root else safe_evaluate(state)
        if score > best_state.score:
            best_state = state
            logger.info("[NEW BEST] score=%.3f at %s", score, state.latex_dir)
        if state.depth >= p.max_depth:
            continue
        if (
            state.is_buggy
            and state.debug_depth < p.max_debug_depth
            and random.random() < p.debug_prob
        ):
            tex_path = state.latex_dir / "template.tex"
            new_src = propose_edit(
                tex_path, seed_ideas, human_reviews, model=model_editor
            )
            tex_path.write_text(new_src)
            state.debug_depth += 1
            safe_evaluate(state)
        # Expand children by proposing edits
        for i in range(p.beam_size):
            child_dir = unique_subdir(state.latex_dir.parent, f"d{state.depth}")
            shutil.copytree(state.latex_dir, child_dir)
            tex_path = child_dir / "template.tex"
            new_source = propose_edit(
                tex_path,
                seed_ideas,
                human_reviews,
                model=model_editor,
            )
            tex_path.write_text(new_source)
            child = PaperNode(
                child_dir,
                state.depth + 1,
                parent=state,
                llm_model=model_review,
                vlm_model=model_vlm,
            )
            state.children.append(child)
            safe_evaluate(child)
            journal.append(child)
            frontier.append(child)
    return journal.best_node(orchestrator_model), journal


def tree_search_improve(
    root_dir: Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    *,
    params: SearchParams | None = None,
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
):
    """Priority-based tree search over paper versions."""
    p = params or SearchParams(max_depth=3, beam_size=4)
    root = PaperNode(root_dir, llm_model=model_review, vlm_model=model_vlm)
    safe_evaluate(root)
    journal = Journal()
    journal.append(root)
    frontier: list[tuple[float, PaperNode]] = [(-root.score, root)]
    for i in range(p.num_drafts):
        draft_dir = unique_subdir(root.latex_dir.parent, "draft")
        shutil.copytree(root.latex_dir, draft_dir)
        tex_path = draft_dir / "template.tex"
        new_source = propose_edit(
            tex_path, seed_ideas, human_reviews, model=model_editor
        )
        tex_path.write_text(new_source)
        draft = PaperNode(
            draft_dir,
            root.depth + 1,
            parent=root,
            llm_model=model_review,
            vlm_model=model_vlm,
        )
        root.children.append(draft)
        draft.evaluate()
        journal.append(draft)
        heapq.heappush(frontier, (-draft.score, draft))

    while frontier:
        _, node = heapq.heappop(frontier)
        logger.info(
            "Exploring depth=%s dir=%s score=%.3f",
            node.depth,
            node.latex_dir,
            node.score,
        )
        if node.depth >= p.max_depth:
            continue
        if (
            node.is_buggy
            and node.debug_depth < p.max_debug_depth
            and random.random() < p.debug_prob
        ):
            tex_path = node.latex_dir / "template.tex"
            new_src = propose_edit(
                tex_path, seed_ideas, human_reviews, model=model_editor
            )
            tex_path.write_text(new_src)
            node.debug_depth += 1
            safe_evaluate(node)
        for i in range(p.beam_size):
            child_dir = unique_subdir(node.latex_dir.parent, f"d{node.depth}")
            shutil.copytree(node.latex_dir, child_dir)
            tex_path = child_dir / "template.tex"
            new_source = propose_edit(
                tex_path,
                seed_ideas,
                human_reviews,
                model=model_editor,
            )
            tex_path.write_text(new_source)
            child = PaperNode(
                child_dir,
                node.depth + 1,
                parent=node,
                llm_model=model_review,
                vlm_model=model_vlm,
            )
            node.children.append(child)
            safe_evaluate(child)
            journal.append(child)
            heapq.heappush(frontier, (-child.score, child))

    return journal.best_node(orchestrator_model), journal
