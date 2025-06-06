"""Breadth-First Tree Search (BFTS) over paper versions."""
from __future__ import annotations

from collections import deque
import heapq
import uuid
from pathlib import Path
import shutil, json
from ai_scientist.treesearch.backend import query, FunctionSpec
from .latex_editor import propose_edit
from .llm_review import llm_review
from .vlm_review import vlm_review
from .meta_review import meta_score

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

    def __init__(self, latex_dir: Path, depth: int = 0, parent: "PaperNode | None" = None):
        self.id = uuid.uuid4().hex
        self.latex_dir = latex_dir
        self.depth = depth
        self.parent = parent
        self.children: list["PaperNode"] = []
        self.pdf_path = latex_dir / "template.pdf"  # compiled later
        self.score: float | None = None
        self.llm_json: dict | None = None
        self.vlm_json: dict | None = None
        # compatibility with treesearch Journal
        self.is_buggy = False
        self.is_buggy_plots = False

    def compile(self):
        from ai_scientist.perform_icbinb_writeup import compile_latex  # reuse util

        compile_latex(str(self.latex_dir), str(self.pdf_path))

    def evaluate(self):
        if not self.pdf_path.exists():
            self.compile()
        self.llm_json = llm_review(str(self.pdf_path))
        self.vlm_json = vlm_review(str(self.pdf_path))
        self.score = meta_score([self.llm_json, self.vlm_json])
        # Persist results for analysis
        with open(self.latex_dir / "reviews.json", "w") as f:
            json.dump({"llm": self.llm_json, "vlm": self.vlm_json, "score": self.score}, f, indent=2)
        return self.score


class Journal:
    """Keep track of explored paper versions."""

    def __init__(self) -> None:
        self.nodes: list[PaperNode] = []

    def append(self, node: PaperNode) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def best_node(self) -> PaperNode | None:
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
                model=ORCHESTRATOR_MODEL,
                temperature=0.3,
            )
            selected = next((n for n in self.nodes if n.id == selection["selected_id"]), None)
            if selected:
                return selected
        except Exception:
            pass
        return max(self.nodes, key=lambda n: n.score or 0)


def breadth_first_improve(
    root_dir: Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    max_depth: int = 3,
    beam_size: int = 4,
):
    """Explore paper edits using best-first strategy."""
    root = PaperNode(root_dir)
    root.evaluate()
    journal = Journal()
    journal.append(root)
    frontier = deque([root])
    best_state = root
    while frontier:
        state = frontier.popleft()
        print(f"Evaluating depth={state.depth} dir={state.latex_dir}")
        score = state.score if state is root else state.evaluate()
        if score > best_state.score:
            best_state = state
            print(f"[NEW BEST] score={score:.3f} at {state.latex_dir}")
        if state.depth >= max_depth:
            continue
        # Expand children by proposing edits
        for i in range(beam_size):
            child_dir = state.latex_dir.parent / f"child_d{state.depth}_{i}"
            shutil.copytree(state.latex_dir, child_dir, dirs_exist_ok=True)
            tex_path = child_dir / "template.tex"
            new_source = propose_edit(tex_path, seed_ideas, human_reviews)
            tex_path.write_text(new_source)
            child = PaperNode(child_dir, state.depth + 1, parent=state)
            state.children.append(child)
            journal.append(child)
            frontier.append(child)
    return journal.best_node(), journal


def tree_search_improve(
    root_dir: Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    max_depth: int = 3,
    beam_size: int = 4,
):
    """Priority-based tree search over paper versions."""
    root = PaperNode(root_dir)
    root.evaluate()
    journal = Journal()
    journal.append(root)
    frontier: list[tuple[float, PaperNode]] = [(-root.score, root)]

    while frontier:
        _, node = heapq.heappop(frontier)
        print(f"Exploring depth={node.depth} dir={node.latex_dir} score={node.score:.3f}")
        if node.depth >= max_depth:
            continue
        for i in range(beam_size):
            child_dir = node.latex_dir.parent / f"node_d{node.depth}_{i}"
            shutil.copytree(node.latex_dir, child_dir, dirs_exist_ok=True)
            tex_path = child_dir / "template.tex"
            new_source = propose_edit(tex_path, seed_ideas, human_reviews)
            tex_path.write_text(new_source)
            child = PaperNode(child_dir, node.depth + 1, parent=node)
            node.children.append(child)
            child.evaluate()
            journal.append(child)
            heapq.heappush(frontier, (-child.score, child))

    return journal.best_node(), journal

