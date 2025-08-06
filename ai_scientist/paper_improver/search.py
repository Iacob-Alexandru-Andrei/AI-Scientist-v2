from __future__ import annotations

from collections.abc import Collection
import concurrent.futures as fut
import logging
import random
import shutil
import uuid
from pathlib import Path

from ai_scientist import llm
from ai_scientist.treesearch.backend import query, FunctionSpec
from .writeup import perform_writeup
from .core import (
    PaperNode,
    Journal,
    safe_evaluate,
    unique_subdir,
    EDITOR_MODEL,
    VLM_MODEL,
    DEFAULT_MODEL,
    ORCHESTRATOR_MODEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# —— GENETIC UTILITY FUNCTIONS ———————————————————————————————————————————
# ---------------------------------------------------------------------------

_SELECTION_SPEC = FunctionSpec(
    name="select_elites",
    description="Return a JSON list of IDs of the best papers",
    json_schema={
        "type": "object",
        "properties": {"elite_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["elite_ids"],
    },
)


def orchestrator_select_elites(
    population: list[PaperNode], k: int, model: str = ORCHESTRATOR_MODEL
) -> list[PaperNode]:
    """Ask the orchestrator model to pick *k* elites from the population."""
    if k <= 0:
        return []

    prompt = {
        "Role": (
            "You are a senior area‑chair.  Select the **best** papers "
            f"(up to {k}) based on the given reviews and scores."
        ),
        "Candidates": "",
    }
    for n in population:
        prompt["Candidates"] += (
            f"ID: {n.id}\n"
            f"Score: {n.score:.3f}\n"
            f"Paper (trimmed):\n{n.tex_path.read_text()[:1_000]}\n"
            f"Reviews: {str(n.llm_jsons)[:1_000]} {str(n.vlm_json)[:500]}\n\n"
        )

    try:
        out = query(
            system_message=prompt,
            user_message=None,
            func_spec=_SELECTION_SPEC,
            model=model,
            temperature=0.3,
        )
        chosen = {cid for cid in out["elite_ids"][:k]}
        return [n for n in population if n.id in chosen][:k]
    except Exception as exc:
        logger.warning(
            "Orchestrator elite selection failed: %s – using top‑k by score", exc
        )
        return sorted(population, key=lambda n: n.score or -1, reverse=True)[:k]


def tournament_pick(
    pop: list[PaperNode],
    tour_size: int,
    selection_model: str,
    *,
    exclude: Collection[PaperNode] | None = None,
) -> PaperNode:
    """
    Return *one* parent chosen via size-`tour_size` tournament selection,
    while omitting any participants listed in *exclude*.

    Parameters
    ----------
    pop : list[PaperNode]
        Current population.
    tour_size : int
        Number of contenders to draw for the tournament.
    selection_model : str
        LLM/VLM to be used by `orchestrator_select_elites`.
    exclude : Collection[PaperNode] | None, optional
        Individuals that must **not** be selected as contenders.

    Raises
    ------
    ValueError
        If fewer than *tour_size* eligible individuals remain after exclusion.
    """
    # 1. Build the eligible pool
    if exclude:
        eligible = [ind for ind in pop if ind not in exclude]
    else:
        eligible = pop

    # 2. Guard against undersized pools
    if len(eligible) < tour_size:
        raise ValueError(
            f"Not enough eligible individuals: need {tour_size}, "
            f"but only {len(eligible)} remain after exclusion."
        )

    # 3. Run the tournament
    contenders = random.sample(eligible, tour_size)
    # `orchestrator_select_elites` returns a list; we need the first (and only) element
    return orchestrator_select_elites(contenders, 1, model=selection_model)[0]


template_instructions = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.
Treat this as the note-taking phase of your review.

In <JSON>, provide the review in JSON format with the following fields in the order justifying where each choice is sourced from (which paper, which review, etc.):
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "General Improvements": A list of general improvements that could be made to the paper.
- "Abstract Improvements": A list of improvements to the abstract.
- "Introduction Improvements": A list of improvements to the introduction.
- "Related Work Improvements": A list of improvements to the related work section.
- "Method Improvements": A list of improvements to the method section.
- "Results Improvements": A list of improvements to the results section.
- "Conclusion Improvements": A list of improvements to the conclusion.
- "References Improvements": A list of improvements to the references.
- "Figures Improvements": A list of improvements to the figures.
- "Math Improvements": A list of improvements to the mathematical formulations that would ensure corectness.

This JSON will be automatically parsed, so ensure the format is precise.
"""

orchestrator_prompt = (
    """
"You are an orchestrator model that serves as a comparative analyzer for papers, your goal is to effectively propose changes to a target paper based on its text, reviews, and the text and reviews of a set of other candidate papers.

This is the target paper:
```
{target_paper}
```

These are its reviews:
```
{target_reviews}
```

These are the candidate papers and their reviews:
```
{candidates}
```

These were the original instructions for the reviews:
```
{form}
```

"""
    + template_instructions
)


def _crossover_papers(
    parent_a: PaperNode, parents: tuple[PaperNode, ...], model: str
) -> str:
    """Use ``model`` to merge positive aspects of ``parent_b`` into ``parent_a``.

    The returned string contains updated LaTeX code for ``parent_a``.
    """

    prompt = (
        "You are an orchestrator model combining improvements from N LaTeX "
        "drafts into a target paper. Propose changes merging the the positive qualities of several Papers B into Paper A without"
        "directly copying text. Return only the recommended changes ```latex"
        " block.\n\n"
        "#### PAPER A\n" + parent_a.tex_path.read_text() + "\n"
    )
    for i, parent_b in enumerate(parents):
        prompt += f"#### PAPER B {i + 1}\n{parent_b.tex_path.read_text()}\n"

    client, m = llm.create_client(model)
    resp = client.chat.completions.create(
        model=m, messages=[{"role": "user", "content": prompt}], temperature=0.7
    )
    print(f"Orchestrator response: {resp.choices[0].message.content[:100]}...")
    return resp.choices[0].message.content


def build_child(
    target: PaperNode,
    parents: Collection[PaperNode],
    *,
    orchestrator_model: str,
    review_model: str,
    vlm_model: str,
    perform_kwargs: dict,
    llm_review_kwargs: dict,
    vlm_review_kwargs: dict,
    journal: Journal,
) -> PaperNode:
    """Create, write‑up and evaluate a new child node."""
    # ------------------------------------------------------------------ merge
    orchestrator_recommendations = _crossover_papers(
        target, tuple(parents), model=orchestrator_model
    )

    success = False
    draft_dir = Path()
    while not success:
        # ------------------------------------------------ create scratch folder
        draft_dir = unique_subdir(
            target.latex_dir.parent.parent, "child_" + uuid.uuid4().hex[:6]
        )
        shutil.copytree(target.latex_dir.parent, draft_dir)
        perform_writeup(
            base_folder=draft_dir,
            model_reviews=str(target.llm_jsons) + str(target.vlm_json),
            recommended_improvements=orchestrator_recommendations,
            human_reviews=human_reviews,
            num_cite_rounds=num_cite_rounds,
            small_model=model_vlm,
            big_model=model_editor,
            n_writeup_reflections=n_writeup_reflections,
            page_limit=page_limit,
        )

    # ------------------------------------------------ wrap as node & evaluate
    child = PaperNode(
        draft_dir / "latex",
        depth=target.depth + 1,
        parent=target,
        llm_model=review_model,
        vlm_model=vlm_model,
    )
    safe_evaluate(child, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
    journal.append(child)
    return child


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# —— MAIN EVOLUTION ROUTINE ———————————————————————————————————————————————
# ---------------------------------------------------------------------------


def genetic_search_improve(
    root_dir: Path,
    *,
    human_reviews: str | None = None,
    num_cite_rounds: int = 2,
    n_writeup_reflections=3,
    population_size: int = 12,
    page_limit: int = 12,
    generations: int = 25,
    elite_size: int = 2,
    tournament_size: int = 3,
    crossover_arity: int = 3,
    model_editor: str = EDITOR_MODEL,
    model_review: str = DEFAULT_MODEL,
    model_vlm: str = VLM_MODEL,
    orchestrator_model: str = ORCHESTRATOR_MODEL,
    elite_selection_model: str = ORCHESTRATOR_MODEL,
    tournament_selection_model: str = ORCHESTRATOR_MODEL,
    perform_kwargs: dict | None = None,
    llm_review_kwargs: dict | None = None,
    vlm_review_kwargs: dict | None = None,
) -> tuple[PaperNode | None, Journal]:
    """
    A steady‑state GA for LaTeX‑paper optimisation.

    • *population_size* : constant |P|.
    • *elite_size*      : number of copies that survive each generation.
    • *tournament_size* : ≥ 2 for parent selection.
    • *crossover_arity* : number of parents to combine into a child, crossover is directional
    • *generations*     : stop after this many full replacements.
    """
    # ---------------- default kwargs ---------------------------------------------------
    perform_kwargs = perform_kwargs or {}
    llm_review_kwargs = llm_review_kwargs or {}
    vlm_review_kwargs = vlm_review_kwargs or {}

    # ---------------- bookkeeping ------------------------------------------------------
    journal = Journal()

    # ---------------- seed population --------------------------------------------------
    root = PaperNode(root_dir, llm_model=model_review, vlm_model=model_vlm)
    safe_evaluate(root, llm_kwargs=llm_review_kwargs, vlm_kwargs=vlm_review_kwargs)
    journal.append(root)

    population: list[PaperNode] = [root]

    while len(population) < population_size:
        success = False
        draft_dir = Path()
        while not success:
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
            if not success:
                logger.warning(
                    "Writeup failed for %s, retrying with a new draft", draft_dir
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
        population.append(draft)  #

    # ---------------- evolutionary loop -----------------------------------------------
    for gen in range(generations):
        logger.info("— Generation %d —", gen + 1)

        # 1. Pick elites
        elites = orchestrator_select_elites(
            population, elite_size, model=elite_selection_model
        )
        logger.debug("Elites: %s", [e.id for e in elites])

        # 2. Create children to fill the remainder of the population
        children: list[PaperNode] = []
        while len(children) < (population_size - elite_size):
            selected: list[PaperNode] = []
            while len(selected) < crossover_arity:
                pick = tournament_pick(
                    population,
                    tournament_size,
                    tournament_selection_model,
                    exclude=set(selected),  # prevents repeats
                )
                selected.append(pick)

            for idx, target in enumerate(selected):
                other_parents = selected[:idx] + selected[idx + 1 :]

                child = build_child(
                    target=target,
                    parents=other_parents,
                    orchestrator_model=orchestrator_model,
                    review_model=model_review,
                    vlm_model=model_vlm,
                    perform_kwargs=perform_kwargs,
                    llm_review_kwargs=llm_review_kwargs,
                    vlm_review_kwargs=vlm_review_kwargs,
                    journal=journal,
                )
                children.append(child)
        # 4. Form next generation
        population = elites + children
        assert len(population) == population_size

    # ---------------- return best found ------------------------------------------------
    best = max(journal.good_nodes, key=lambda n: n.score or -1, default=None)
    return best, journal
