# AGENT Notes

This repository implements the **AI Scientist v2** project. It contains tools for automated scientific discovery via agentic tree search and includes a minimal `paper_improver` subpackage for iteratively refining LaTeX papers.

Below is an overview of the repository structure with short descriptions of the most relevant modules and scripts. This file is intended for Codex agents working on the repository.

## Top-Level Layout

```
AI-Scientist-v2/
├── ai_scientist/           # Main Python package
├── docs/                   # Documentation assets
├── examples/               # Example data for quick runs
├── scripts/                # Command line entry points
├── tests/                  # Pytest suite
├── launch_scientist_bfts.py# Full experiment launcher
├── bfts_config.yaml        # Default config for tree search
└── README.md               # User-facing instructions
```

### Root Scripts
- **`launch_scientist_bfts.py`** – The original entry point for tree-search experiments. It orchestrates ideation, experiment execution, plotting, write-up generation, and peer reviews. It relies heavily on modules under `ai_scientist/treesearch`.
- **`scripts/launch_paper_improver.py`** – Simplified CLI for the `paper_improver` package. It accepts paths to a LaTeX project, seed ideas, optional human reviews, and various model names/keys.

## `ai_scientist/` Package
This package houses all functionality. Notable submodules include:

- **`llm.py`** – Helpers to create clients for OpenAI, Gemini, or Claude models and wrappers for chat completions.
- **`vlm.py`** – Utilities for vision-language models and image handling.
- **`perform_ideation_temp_free.py`** – Generates research ideas from a Markdown prompt using LLMs and the Semantic Scholar tool.
- **`perform_icbinb_writeup.py`** and **`perform_writeup.py`** – LLM-driven LaTeX generation utilities. They also contain `compile_latex` for rendering PDFs.
- **`perform_llm_review.py`** and **`perform_vlm_review.py`** – Modules to review papers or code with LLM/VLM models.
- **`tools/`** – Contains utilities such as `semantic_scholar.py` for literature search.
- **`treesearch/`** – The main agentic tree-search implementation. It defines the `Journal`, `Node`, backends for LLM calls, and utilities for exploring code versions. The `perform_experiments_bfts_with_agentmanager.py` module launches the full search process.
- **`paper_improver/`** – The new package providing a minimal pipeline for improving LaTeX papers without running experiments (described in detail below).

## `paper_improver` Subpackage
This folder contains a simplified pipeline that reuses LLM/VLM review utilities and the tree-search style search loop. Key modules:

- **`__init__.py`** – Re-exports `improve_paper` for convenience.
- **`latex_editor.py`** – Uses an LLM to propose and apply edits to a LaTeX file, extracting updated code from a fenced ` ```latex` block.
- **`llm_review.py`** / **`vlm_review.py`** – Thin wrappers around `perform_llm_review.perform_review` and `perform_vlm_review.perform_imgs_cap_ref_review`.
- **`meta_review.py`** – Aggregates multiple review JSON objects into a single numerical score.
- **`search.py`** – Implements two search strategies:
  - `breadth_first_improve` – Basic breadth-first search over paper versions.
  - `tree_search_improve` – Priority-based search mirroring the main repository’s tree search but omitting experiment execution.
  - Defines `PaperNode`, `Journal`, and `ORCHESTRATOR_MODEL` for orchestrated selection of the best node.
- **`pipeline.py`** – High-level function `improve_paper` that chooses the search strategy and passes model names along.
- **`utils.py`** – Helper functions like `unique_subdir` for creating non-colliding directories.

The `examples/paper_improver_minimal/` directory includes a sample LaTeX project, seed ideas JSON, and human reviews to demonstrate the pipeline.

## Testing
The `tests/` directory provides a small pytest suite. `conftest.py` stubs heavy dependencies (LLM clients, token tracker, tree-search backend) so that tests run quickly offline. Tests cover `meta_review`, the pipeline’s strategy selection, and core search logic by patching out network calls.

To run all tests:
```bash
pytest -q
```

## Development Tips
- Most modules rely on environment variables for API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.). The CLI scripts also allow passing keys as arguments.
- Tree search results (when using the full `launch_scientist_bfts.py` pipeline) are stored under `experiments/` in timestamped folders with logs and HTML visualizations.
- The repository’s Python code is formatted with `black` and tests assume Python 3.11.

This overview should help orient new Codex agents when making modifications or adding features.
