"""Helper utilities for the :mod:`paper_improver` package.

Currently this module only contains ``unique_subdir`` which generates a
non-colliding directory name.  It is primarily used by the search routines to
clone LaTeX templates into new working folders.
"""

from __future__ import annotations

from pathlib import Path
import uuid
from datetime import datetime
import os
import os.path as osp
import re


def unique_subdir(parent: Path, prefix: str) -> Path:
    """Return a subdirectory path within *parent* that does not already exist."""

    while True:
        # Include timestamp so folders are ordered chronologically when listed
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Generate a random name and check if it exists.  The directory is not
        # created here; callers may decide whether to copy or symlink data.
        cand = parent / f"{prefix}_{uuid.uuid4().hex[:8]}_{ts}"
        if not cand.exists():
            return cand


def find_pdf_path_for_review(idea_dir: str | Path) -> str:
    """Return the most recent reflection PDF within *idea_dir*."""

    if not os.path.isdir(idea_dir):
        return ""

    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            return osp.join(idea_dir, final_pdfs[0])
        reflection_nums = []
        for f in reflection_pdfs:
            match = re.search(r"reflection[_.]?(\d+)", f)
            if match:
                reflection_nums.append((int(match.group(1)), f))
        if reflection_nums:
            highest = max(reflection_nums, key=lambda x: x[0])
            return osp.join(idea_dir, highest[1])
        return osp.join(idea_dir, reflection_pdfs[0])
    return osp.join(idea_dir, pdf_files[0]) if pdf_files else ""
