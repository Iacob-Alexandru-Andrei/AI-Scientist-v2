#!/usr/bin/env python3
"""CLI for the paper improver."""
import argparse, json
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
args = parser.parse_args()

seed_ideas = json.loads(Path(args.seed_ideas_json).read_text())
human_reviews = Path(args.human_reviews).read_text() if args.human_reviews else None

improve_paper(
    args.latex_dir,
    json.dumps(seed_ideas, indent=2),
    human_reviews,
    max_depth=args.max_depth,
    beam_size=args.beam_size,
    strategy=args.strategy,
)
