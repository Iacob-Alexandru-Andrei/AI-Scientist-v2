# Minimal Paper Improver Example

This folder contains a tiny LaTeX project and sample inputs to run the
`paper_improver` pipeline. The example is intentionally simple and uses a
single LaTeX source file with one bibliography entry.

```bash
python scripts/launch_paper_improver.py examples/paper_improver_minimal \
    examples/paper_improver_minimal/seed_ideas.json \
    --human-reviews examples/paper_improver_minimal/human_reviews.txt \
    --max-depth 1 --beam-size 1 \
    --model-editor o1-preview-2024-09-12 \
    --model-review gpt-4o-2024-11-20
```
