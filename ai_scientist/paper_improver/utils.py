"""Helper utilities for the :mod:`paper_improver` package.

Currently this module only contains ``unique_subdir`` which generates a
non-colliding directory name.  It is primarily used by the search routines to
clone LaTeX templates into new working folders.
"""

from __future__ import annotations

from pathlib import Path
import uuid


def unique_subdir(parent: Path, prefix: str) -> Path:
    """Return a subdirectory path within *parent* that does not already exist."""

    while True:
        # Generate a random name and check if it exists.  The directory is not
        # created here; callers may decide whether to copy or symlink data.
        cand = parent / f"{prefix}_{uuid.uuid4().hex[:8]}"
        if not cand.exists():
            return cand
