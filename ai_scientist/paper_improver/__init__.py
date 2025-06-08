"""Top-level helpers for the :mod:`paper_improver` package.

This module exposes a small public surface consisting of the
``improve_paper`` pipeline function and ``reflect_paper`` utility.  The
actual implementation lives in the submodules under ``paper_improver``.

The rest of the package mirrors the larger AI-Scientist tree search but is
specialised for iteratively polishing LaTeX papers.  ``improve_paper``
handles high level orchestration while ``reflect_paper`` runs a final
polishing loop using ``chktex`` and vision-language reviews.
"""

from .pipeline import improve_paper
from .reflection import reflect_paper

__all__ = ["improve_paper", "reflect_paper"]
