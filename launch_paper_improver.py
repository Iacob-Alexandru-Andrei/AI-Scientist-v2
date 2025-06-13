#!/usr/bin/env python3
"""Command line entry point for ``paper_improver``.

This script parses command line arguments, configures logging and environment
variables and then invokes :func:`ai_scientist.paper_improver.improve_paper`.
It mirrors the ``launch_scientist_bfts.py`` script from the main project but
omits experiment execution.
"""

import argparse
import json
import logging
from pathlib import Path
from random import seed

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None
from ai_scientist.paper_improver import improve_paper

parser = argparse.ArgumentParser(
    description="Iteratively improve an existing paper via AI-Scientist pipeline"
)
parser.add_argument("latex_dir", help="Directory containing template.tex (and figures)")
parser.add_argument("--research_idea", help="MD file with high-level improvement ideas")
parser.add_argument(
    "--human-reviews", help="Path to txt/markdown file with reviewer comments"
)
parser.add_argument("--max-depth", type=int, default=2)
parser.add_argument("--beam-size", type=int, default=3)
parser.add_argument("--num-drafts", type=int, default=3)
parser.add_argument("--debug-prob", type=float, default=0.5)
parser.add_argument("--max-debug-depth", type=int, default=3)
parser.add_argument(
    "--strategy",
    choices=["bfs", "tree"],
    default="bfs",
    help="Search strategy to use: simple bfs or priority tree search",
)
parser.add_argument(
    "--model-editor",
    default="gemini-2.5-flash-preview-04-17",
    help="LLM used to propose edits",
)
parser.add_argument(
    "--model-review",
    default="gemini-2.5-flash-preview-04-17",
    help="Model used for text-based review",
)
parser.add_argument(
    "--model-vlm",
    default="gemini-2.5-flash-preview-04-17",
    help="Model used for VLM figure review",
)
parser.add_argument(
    "--model-orchestrator",
    default="gemini-2.5-flash-preview-04-17",
    help="Model used to select the best node",
)
parser.add_argument(
    "--model-citation",
    default="gemini-2.5-flash-preview-04-17",
    help="Model used for citation gathering",
)
parser.add_argument(
    "--num-cite-rounds",
    type=int,
    default=20,
    help="Number of citation rounds",
)
parser.add_argument(
    "--model-reflection",
    default="gemini-2.5-flash-preview-04-17",
    help="Model used for final reflection",
)
parser.add_argument(
    "--num-reflections",
    type=int,
    default=3,
    help="Number of reflection steps",
)
parser.add_argument(
    "--page-limit",
    type=int,
    default=4,
    help="Page limit for final reflection",
)
parser.add_argument(
    "--num-reviewers",
    type=int,
    default=1,
    help="Number of reviewers for the final review",
)
parser.add_argument("--llm-num-reflections", type=int, default=1)
parser.add_argument("--llm-num-fs-examples", type=int, default=1)
parser.add_argument("--llm-temperature", type=float, default=0.75)
parser.add_argument(
    "--output-dir",
    default=None,
    help="Optional directory to copy the project and outputs",
)
parser.add_argument(
    "--config",
    default="paper_improver_config.yaml",
    help="Path to YAML config with default values",
)
parser.add_argument("--openai-api-key")
parser.add_argument("--gemini-api-key")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Load defaults from YAML config if available
config_path = Path(args.config)
if yaml is not None and config_path.exists():
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = cfg.get("paper_improver", {})

    def apply_default(attr, key=None, conf=cfg):
        if getattr(args, attr) == parser.get_default(attr) and key in conf:
            setattr(args, attr, conf[key])

    apply_default("strategy", "strategy")
    apply_default("max_depth", "max_depth")
    apply_default("beam_size", "beam_size")
    apply_default("num_drafts", "num_drafts")
    apply_default("debug_prob", "debug_prob")
    apply_default("max_debug_depth", "max_debug_depth")
    apply_default("num_cite_rounds", "num_cite_rounds")
    apply_default("num_reflections", "num_reflections")
    apply_default("page_limit", "page_limit")

    llm_cfg = cfg.get("llm_review", {})
    apply_default("llm_num_reflections", key="num_reflections", conf=llm_cfg)
    apply_default("num_reviewers", "num_reviews_ensemble", conf=llm_cfg)
    apply_default("llm_num_fs_examples", key="num_fs_examples", conf=llm_cfg)
    apply_default("llm_temperature", key="temperature", conf=llm_cfg)
    apply_default("output_dir", "output_dir")

    models = cfg.get("models", {})
    for field in [
        ("model_editor", "model_editor"),
        ("model_review", "model_review"),
        ("model_vlm", "model_vlm"),
        ("model_orchestrator", "model_orchestrator"),
        ("model_citation", "model_citation"),
        ("model_reflection", "model_reflection"),
    ]:
        attr, key = field
        if getattr(args, attr) == parser.get_default(attr) and key in models:
            setattr(args, attr, models[key])

# Allow API keys to be supplied via command line to avoid modifying the shell
if args.openai_api_key:
    import os

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
if args.gemini_api_key:
    import os

    os.environ["GEMINI_API_KEY"] = args.gemini_api_key

human_reviews = Path(args.human_reviews).read_text()
seed_ideas = Path(args.research_idea).read_text()

improve_paper(
    args.latex_dir,
    seed_ideas,
    human_reviews=human_reviews,
    max_depth=args.max_depth,
    beam_size=args.beam_size,
    num_drafts=args.num_drafts,
    debug_prob=args.debug_prob,
    max_debug_depth=args.max_debug_depth,
    strategy=args.strategy,
    model_editor=args.model_editor,
    model_review=args.model_review,
    model_vlm=args.model_vlm,
    orchestrator_model=args.model_orchestrator,
    model_citation=args.model_citation,
    num_cite_rounds=args.num_cite_rounds,
    model_reflection=args.model_reflection,
    num_reflections=args.num_reflections,
    page_limit=args.page_limit,
    num_reviewers=args.num_reviewers,
    llm_num_reflections=args.llm_num_reflections,
    llm_num_fs_examples=args.llm_num_fs_examples,
    llm_temperature=args.llm_temperature,
    output_dir=args.output_dir,
)  # run the search pipeline
