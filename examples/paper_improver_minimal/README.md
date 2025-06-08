# Minimal Paper Improver Example

This directory contains a tiny LaTeX project with a placeholder figure, a
bibliography, seed improvement ideas and reviewer comments. It can be used to
try the `paper_improver` pipeline without extra setup.

Example command:

```bash
python scripts/launch_paper_improver.py examples/paper_improver_minimal \
    examples/paper_improver_minimal/seed_ideas.json \
    --human-reviews examples/paper_improver_minimal/human_reviews.txt \
    --max-depth 1 --beam-size 1 --num-drafts 1 \
    --debug-prob 0.5 --max-debug-depth 3 \
    --model-editor o1-preview-2024-09-12 \
    --model-review gpt-4o-2024-11-20 \
    --model-vlm gpt-4o-2024-11-20 \
    --model-orchestrator gpt-4o-2024-11-20 \
    --model-citation gpt-4o-2024-11-20 --num-cite-rounds 20 \
    --model-reflection o1-preview-2024-09-12 --num-reflections 1 --page-limit 4
```
