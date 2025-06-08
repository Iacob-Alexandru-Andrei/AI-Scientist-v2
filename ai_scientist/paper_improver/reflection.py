"""Utility for post-processing the best paper version.

The ``reflect_paper`` function repeatedly compiles the LaTeX source and feeds
the resulting PDF through ``chktex`` and a VLM based figure review.  The output
of these tools is provided to an editing model which returns an updated LaTeX
snippet.  The loop terminates early if no changes are made or if the page limit
is exceeded.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import textwrap
from pathlib import Path

from ai_scientist.llm import create_client
from ai_scientist.perform_icbinb_writeup import (
    compile_latex,
    get_reflection_page_info,
)
from ai_scientist.perform_vlm_review import (
    perform_imgs_cap_ref_review,
    detect_duplicate_figures,
)
from ai_scientist.vlm import create_client as create_vlm_client

logger = logging.getLogger(__name__)

REFLECTION_MODEL = "o1-preview-2024-09-12"
DEFAULT_ROUNDS = 3
DEFAULT_PAGE_LIMIT = 4


def reflect_paper(
    latex_dir: Path,
    *,
    model: str = REFLECTION_MODEL,
    vlm_model: str = "gpt-4o-2024-11-20",
    num_rounds: int = DEFAULT_ROUNDS,
    page_limit: int = DEFAULT_PAGE_LIMIT,
) -> None:
    """Run a reflection loop over the LaTeX source to polish the paper."""

    tex_path = latex_dir / "template.tex"  # file that will be repeatedly updated
    # Create clients for the LLM and VLM models used in this step
    client, m = create_client(model)
    vlm_client, vm = create_vlm_client(vlm_model)
    msg_history: list[dict] = []

    for i in range(num_rounds):
        # Compile the current LaTeX source to PDF for analysis
        pdf_path = latex_dir / f"reflection_{i + 1}.pdf"
        compile_latex(str(latex_dir), str(pdf_path))

        try:
            # ``chktex`` catches common LaTeX style issues. We ignore a few
            # checks that produce noisy suggestions.
            res = subprocess.run(
                ["chktex", str(tex_path), "-q", "-n2", "-n24", "-n13", "-n1"],
                capture_output=True,
                text=True,
            )
            chk_output = res.stdout
        except Exception as exc:
            logger.error("chktex failed: %s", exc)
            chk_output = ""

        # Vision-language model checks ensure figures and captions are coherent
        vlm_rev = perform_imgs_cap_ref_review(vlm_client, vm, str(pdf_path))
        dup_figs = detect_duplicate_figures(vlm_client, vm, str(pdf_path))
        page_info = get_reflection_page_info(str(pdf_path), page_limit)

        prompt = textwrap.dedent(
            f"""
            Now let's reflect on the paper and identify any issues.
            {page_info}
            chktex results:
            ```
            {chk_output}
            ```
            VLM review:
            ```
            {vlm_rev}
            ```
            Duplicate figures:
            ```
            {dup_figs}
            ```
            Provide the revised LaTeX in a fenced ```latex block.
            """
        )

        # Ask the LLM to produce an improved LaTeX snippet given the diagnostics
        msg_history.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=m,
            messages=msg_history,
            temperature=0.3,
        )
        msg = resp.choices[0].message
        role = getattr(msg, "role", "assistant")
        content = getattr(msg, "content", "")
        msg_history.append({"role": role, "content": content})
        code = re.search(
            r"```latex\s*(.*?)```", resp.choices[0].message.content, re.DOTALL
        )
        if not code:
            break
        new_src = textwrap.dedent(code.group(1)).strip()
        current = tex_path.read_text()
        if new_src == current:
            # Stop if the model produced identical LaTeX; nothing left to refine
            break
        tex_path.write_text(new_src)

    # Finally compile once more so the directory contains the latest PDF
    compile_latex(str(latex_dir), str(latex_dir / "template.pdf"))
