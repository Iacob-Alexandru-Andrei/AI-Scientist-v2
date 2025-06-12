from __future__ import annotations

"""Parallel tree search for LaTeX paper improvements.

This module mirrors :mod:`ai_scientist.treesearch.parallel_agent` but removes
any experiment execution logic. Each worker simply proposes an edit and
computes the LLM/VLM score for the resulting PDF. The search expands nodes in
parallel to speed up exploration.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import logging
import random
from typing import Iterable, List, Tuple

from .search import (
    PaperNode,
    Journal,
    SearchParams,
    safe_evaluate,
    propose_edit,
    EDITOR_MODEL,
    DEFAULT_MODEL,
    VLM_MODEL,
    ORCHESTRATOR_MODEL,
)
from .utils import unique_subdir
from .writeup import perform_writeup

logger = logging.getLogger(__name__)


def _expand_and_score(
    args: Tuple[
        PaperNode,
        str,
        str | None,
        int,
        str,
        str,
        str,
        dict | None,
        dict | None,
        dict | None,
    ],
):
    (
        node,
        seed_ideas,
        human_reviews,
        beam_size,
        model_editor,
        model_review,
        model_vlm,
        writeup_params,
        llm_kwargs,
        vlm_kwargs,
    ) = args
    children = []
    for _ in range(beam_size):
        child_dir = unique_subdir(node.latex_dir.parent, f"d{node.depth}")
        shutil.copytree(node.latex_dir, child_dir)
        tex_path = child_dir / "template.tex"
        new_source = propose_edit(
            tex_path,
            seed_ideas,
            model_reviews=str(node.llm_json) + str(node.vlm_json),
            human_reviews=human_reviews,
            model=model_editor,
        )
        tex_path.write_text(new_source)
        if writeup_params is not None:
            perform_writeup(child_dir, **writeup_params)
        child = PaperNode(
            child_dir,
            node.depth + 1,
            parent=node,
            llm_model=model_review,
            vlm_model=model_vlm,
        )
        node.children.append(child)
        safe_evaluate(child, llm_kwargs=llm_kwargs, vlm_kwargs=vlm_kwargs)
        children.append(child)
    return children


def _get_leaves(node: PaperNode) -> List[PaperNode]:
    """Return all leaf nodes under ``node``."""
    if not node.children:
        return [node]
    leaves: List[PaperNode] = []
    for child in node.children:
        leaves.extend(_get_leaves(child))
    return leaves


def _select_parallel_nodes(
    journal: Journal, params: SearchParams, num_workers: int
) -> List[PaperNode | None]:
    """Mimic the clever node selection from the main project."""
    nodes_to_process: List[PaperNode | None] = []
    processed_trees: set[int] = set()

    while len(nodes_to_process) < num_workers:
        # drafting phase
        if len(journal.draft_nodes) < params.num_drafts:
            nodes_to_process.append(None)
            continue

        viable_trees = [
            root
            for root in journal.draft_nodes
            if not all(leaf.is_buggy for leaf in _get_leaves(root))
        ]

        # try debugging buggy leaves occasionally
        if random.random() < params.debug_prob:
            debuggable = [
                n
                for n in journal.buggy_nodes
                if n.is_buggy
                and n.children == []
                and n.debug_depth <= params.max_debug_depth
            ]
            if debuggable:
                node = random.choice(debuggable)
                tree_root = node
                while tree_root.parent:
                    tree_root = tree_root.parent
                tree_id = id(tree_root)
                if tree_id not in processed_trees or len(processed_trees) >= len(
                    viable_trees
                ):
                    nodes_to_process.append(node)
                    processed_trees.add(tree_id)
                    continue

        # normal improvement selection
        good_nodes = journal.good_nodes
        if not good_nodes:
            nodes_to_process.append(None)
            continue

        best_node = journal.best_node()
        tree_root = best_node
        while tree_root.parent:
            tree_root = tree_root.parent
        tree_id = id(tree_root)
        if tree_id not in processed_trees or len(processed_trees) >= len(viable_trees):
            nodes_to_process.append(best_node)
            processed_trees.add(tree_id)
            continue

        for node in sorted(good_nodes, key=lambda n: n.score or 0, reverse=True):
            tree_root = node
            while tree_root.parent:
                tree_root = tree_root.parent
            tree_id = id(tree_root)
            if tree_id not in processed_trees or len(processed_trees) >= len(
                viable_trees
            ):
                nodes_to_process.append(node)
                processed_trees.add(tree_id)
                break

    return nodes_to_process


def parallel_tree_search_improve(
    root_dir: Path,
    seed_ideas: str,
    human_reviews: str | None = None,
    *,
    params: SearchParams | None = None,
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
    max_workers: int = 4,
    writeup_params: dict | None = None,
    llm_review_kwargs: dict | None = None,
    vlm_review_kwargs: dict | None = None,
) -> Tuple[PaperNode, Journal]:
    """Run a simple parallel tree search over LaTeX edits."""
    p = params or SearchParams(max_depth=3, beam_size=4)
    if writeup_params is None:
        writeup_params = p.writeup_params
    if llm_review_kwargs is None:
        llm_review_kwargs = p.llm_review_kwargs or {}
    if vlm_review_kwargs is None:
        vlm_review_kwargs = p.vlm_review_kwargs or {}
    root = PaperNode(root_dir, llm_model=model_review, vlm_model=model_vlm)
    safe_evaluate(root, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
    journal = Journal()
    journal.append(root)
    best_state = root

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        depth = 0
        # keep iterating until max depth reached
        while depth < p.max_depth:
            nodes = _select_parallel_nodes(journal, p, max_workers)
            tasks = []
            for node in nodes:
                parent = node or root
                tasks.append(
                    ex.submit(
                        _expand_and_score,
                        (
                            parent,
                            seed_ideas,
                            human_reviews,
                            p.beam_size,
                            model_editor,
                            model_review,
                            model_vlm,
                            writeup_params,
                            llm_review_kwargs,
                            vlm_review_kwargs,
                        ),
                    )
                )

            new_nodes: List[PaperNode] = []
            for fut in tasks:
                for child in fut.result():
                    journal.append(child)
                    parent = child.parent
                    if parent:
                        parent.children.append(child)
                    if best_state.score is None or (
                        child.score
                        and best_state.score
                        and child.score > best_state.score
                    ):
                        best_state = child
                    new_nodes.append(child)
            if not new_nodes:
                break
            depth = max(n.depth for n in new_nodes)

    return journal.best_node(orchestrator_model), journal
