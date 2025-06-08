#!/usr/bin/env python3
"""CLI for the paper improver."""
import argparse
import json
import logging
from pathlib import Path
from ai_scientist.paper_improver import improve_paper

parser = argparse.ArgumentParser(description="Iteratively improve an existing paper via AI-Scientist pipeline")
parser.add_argument("latex_dir", help="Directory containing template.tex (and figures)")
parser.add_argument("seed_ideas_json", help="JSON file with high-level improvement ideas")
parser.add_argument("--human-reviews", help="Path to txt/markdown file with reviewer comments")
parser.add_argument("--max-depth", type=int, default=2)
parser.add_argument("--beam-size", type=int, default=3)
parser.add_argument(
    "--strategy",
    choices=["bfs", "tree"],
    default="bfs",
    help="Search strategy to use: simple bfs or priority tree search",
)
parser.add_argument(
    "--model-editor",
    default="o1-preview-2024-09-12",
    help="LLM used to propose edits",
)
parser.add_argument(
    "--model-review",
    default="gpt-4o-2024-11-20",
    help="Model used for text-based review",
)
parser.add_argument(
    "--model-vlm",
    default="gpt-4o-2024-11-20",
    help="Model used for VLM figure review",
)
parser.add_argument(
    "--model-orchestrator",
    default="gpt-4o-2024-11-20",
    help="Model used to select the best node",
)
parser.add_argument("--openai-api-key")
parser.add_argument("--gemini-api-key")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

if args.openai_api_key:
    import os

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
if args.gemini_api_key:
    import os

    os.environ["GEMINI_API_KEY"] = args.gemini_api_key

seed_ideas = Path(args.seed_ideas_json).read_text()
human_reviews = Path(args.human_reviews).read_text() if args.human_reviews else None

improve_paper(
    args.latex_dir,
    seed_ideas,
    human_reviews,
    max_depth=args.max_depth,
    beam_size=args.beam_size,
    strategy=args.strategy,
    model_editor=args.model_editor,
    model_review=args.model_review,
    model_vlm=args.model_vlm,
    orchestrator_model=args.model_orchestrator,
)
