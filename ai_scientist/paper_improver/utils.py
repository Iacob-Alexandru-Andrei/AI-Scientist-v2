"""Helper utilities for the :mod:`paper_improver` package."""

from __future__ import annotations

from pathlib import Path
import uuid


def unique_subdir(parent: Path, prefix: str) -> Path:
    """Return a subdirectory path within *parent* that does not already exist."""

    while True:
        cand = parent / f"{prefix}_{uuid.uuid4().hex[:8]}"
        if not cand.exists():
            return cand
