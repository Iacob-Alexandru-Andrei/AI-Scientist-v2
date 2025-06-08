"""Writeup utilities for the paper improver pipeline."""

from __future__ import annotations

import os
import json
import shutil
import re
import traceback
from pathlib import Path

try:
    from ai_scientist.perform_icbinb_writeup import (
        compile_latex,
        get_reflection_page_info,
        writeup_system_message_template,
        writeup_prompt,
        gather_citations,
    )
    from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
    from ai_scientist.llm import create_client, get_response_from_llm
    from ai_scientist.vlm import create_client as create_vlm_client
except Exception:  # pragma: no cover - fallback when dependencies missing
    compile_latex = lambda *a, **k: None
    get_reflection_page_info = lambda *a, **k: ""
    writeup_system_message_template = "Write a paper"
    writeup_prompt = "{latex_writeup}"
    gather_citations = lambda *a, **k: ""
    perform_imgs_cap_ref_review = lambda *a, **k: {}
    def create_client(model):
        class Dummy:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        class R:
                            message = type("M", (), {"content": ""})

                        return type("Resp", (), {"choices": [R()]})

        return Dummy(), model

    def create_vlm_client(model):
        return create_client(model)
    def get_response_from_llm(*a, **k):
        return ("", [])


def perform_writeup(
    base_folder: str | Path,
    citations_text: str | None = None,
    *,
    no_writing: bool = False,
    num_cite_rounds: int = 20,
    small_model: str = "gpt-4o-2024-05-13",
    big_model: str = "o1-2024-12-17",
    n_writeup_reflections: int = 3,
    page_limit: int = 4,
) -> bool:
    """Generate a polished PDF for the given LaTeX folder.

    This is a simplified variant of ``perform_writeup`` used by the main
    AI-Scientist pipeline. It omits research idea context and focuses solely on
    improving the supplied LaTeX project.
    """

    base_folder = Path(base_folder)
    pdf_file = base_folder / f"{base_folder.name}.pdf"
    latex_folder = base_folder / "latex"

    if latex_folder.exists():
        shutil.rmtree(latex_folder)
    if pdf_file.exists():
        pdf_file.unlink()

    if citations_text is None:
        citations_text = gather_citations(
            base_folder,
            num_cite_rounds=num_cite_rounds,
            small_model=small_model,
        )
        if citations_text is None:
            citations_text = ""

    try:
        shutil.copytree(
            "ai_scientist/blank_icml_latex", latex_folder, dirs_exist_ok=True
        )
        writeup_file = latex_folder / "template.tex"
        tex_content = writeup_file.read_text()
        if citations_text:
            tex_content = tex_content.replace(
                "\\end{filecontents}",
                f"\n{citations_text}\n\\end{filecontents}",
            )
            writeup_file.write_text(tex_content)

        if no_writing:
            compile_latex(str(latex_folder), str(pdf_file))
            return pdf_file.exists()

        client, small_m = create_client(small_model)
        big_client, big_m = create_client(big_model)

        msg = f"Improve the following LaTeX paper. Return full updated code in a fenced block.\n```latex\n{tex_content}\n```"
        resp = client.chat.completions.create(
            model=small_m,
            messages=[{"role": "user", "content": msg}],
            temperature=0.4,
        )
        code_match = re.search(
            r"```latex(.*?)```", resp.choices[0].message.content, re.DOTALL
        )
        if code_match:
            tex_content = code_match.group(1).strip()
            writeup_file.write_text(tex_content)

        big_sys_msg = writeup_system_message_template.format(page_limit=page_limit)
        combined_prompt = writeup_prompt.format(
            idea_text="",
            summaries="",
            aggregator_code="",
            plot_list="",
            latex_writeup=tex_content,
            plot_descriptions="",
        )
        response, msg_history = get_response_from_llm(
            combined_prompt,
            client=big_client,
            model=big_m,
            system_message=big_sys_msg,
            print_debug=False,
        )
        latex_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if latex_match:
            tex_content = latex_match.group(1).strip()
            writeup_file.write_text(tex_content)

        for i in range(n_writeup_reflections):
            reflection_pdf = base_folder / f"reflection_{i+1}.pdf"
            compile_latex(str(latex_folder), str(reflection_pdf))
            vlm_client, vm = create_vlm_client("gpt-4o-2024-05-13")
            img_rev = perform_imgs_cap_ref_review(vlm_client, vm, str(reflection_pdf))
            page_info = get_reflection_page_info(str(reflection_pdf), page_limit)
            prompt = (
                f"Reflect on the paper. Page info: {page_info}\nVLM review:\n```{img_rev}```\n"
                "Return updated LaTeX in a fenced ```latex``` block if changes are needed."
            )
            reflection_resp, msg_history = get_response_from_llm(
                prompt,
                client=big_client,
                model=big_m,
                system_message=big_sys_msg,
                msg_history=msg_history,
                print_debug=False,
            )
            m = re.search(r"```latex(.*?)```", reflection_resp, re.DOTALL)
            if not m:
                break
            new_src = m.group(1).strip()
            cur = writeup_file.read_text()
            if new_src == cur:
                break
            writeup_file.write_text(new_src)

        compile_latex(str(latex_folder), str(pdf_file))
        return pdf_file.exists()

    except Exception:
        print("EXCEPTION in perform_writeup:")
        print(traceback.format_exc())
        return False
