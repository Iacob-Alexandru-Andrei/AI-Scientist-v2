# Minimal Paper Improver Example

This directory contains a tiny LaTeX project with a placeholder figure, a
bibliography, seed improvement ideas and reviewer comments. It can be used to
try the `paper_improver` pipeline without extra setup.

Example command:

```bash
uv run python launch_paper_improver.py examples/paper_improver_minimal \
    examples/paper_improver_minimal/seed_ideas.json \
    --human-reviews examples/paper_improver_minimal/human_reviews.txt \
    --max-depth 1 --beam-size 1 --num-drafts 1 \
    --debug-prob 0.5 --max-debug-depth 3 \
    --model-editor gemini-2.5-flash-preview-04-17 \
    --model-review gemini-2.5-flash-preview-04-17 \
    --model-vlm gpt-4o-2024-11-20 \
    --model-orchestrator gemini-2.5-flash-preview-04-17 \
    --model-citation gemini-2.5-flash-preview-04-17 --num-cite-rounds 20 \
    --model-reflection gemini-2.5-flash-preview-04-17 --num-reflections 1 --page-limit 4
```
