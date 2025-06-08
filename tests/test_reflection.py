from pathlib import Path
import types
from ai_scientist.paper_improver import reflection


def test_reflect_paper(monkeypatch, tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    tex = root / "template.tex"
    tex.write_text("A")

    def fake_compile(cwd, pdf):
        Path(pdf).write_text(Path(cwd).joinpath("template.tex").read_text())

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="```latex\n" + tex.read_text() + "B\n```"))]
                    )

    monkeypatch.setattr(reflection, "compile_latex", fake_compile)
    monkeypatch.setattr(reflection, "create_client", lambda m: (DummyClient(), m))
    monkeypatch.setattr(reflection, "create_vlm_client", lambda m: (object(), m))
    monkeypatch.setattr(reflection, "perform_imgs_cap_ref_review", lambda *a, **k: {})
    monkeypatch.setattr(reflection, "detect_duplicate_figures", lambda *a, **k: {})
    monkeypatch.setattr(reflection, "get_reflection_page_info", lambda *a, **k: "")
    monkeypatch.setattr(reflection.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))

    reflection.reflect_paper(root, num_rounds=1, model="m", vlm_model="v", page_limit=4)
    assert tex.read_text().endswith("B")
