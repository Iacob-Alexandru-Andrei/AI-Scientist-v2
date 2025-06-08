import types
from pathlib import Path
import pytest

from ai_scientist.paper_improver import search, reflection, meta_review


def test_safe_evaluate_marks_buggy(monkeypatch, tmp_path):
    node = search.PaperNode(tmp_path)

    def bad_eval(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(search.PaperNode, "evaluate", bad_eval)
    res = search.safe_evaluate(node)
    assert res is None
    assert node.is_buggy


def test_journal_best_node_fallback(monkeypatch):
    j = search.Journal()
    n1 = search.PaperNode(Path("p1"))
    n2 = search.PaperNode(Path("p2"))
    n1.score = 1.0
    n2.score = 2.0
    j.append(n1)
    j.append(n2)

    def boom(**kw):
        raise RuntimeError("fail")

    monkeypatch.setattr(search, "query", boom)
    best = j.best_node("model")
    assert best is n2  # falls back to numeric max


def test_meta_score_missing_keys():
    review = {"Overall": 6}
    score = meta_review.score_single(review)
    assert score > 0
    assert meta_review.meta_score([review]) == pytest.approx(score)


def test_reflect_paper_early_exit(monkeypatch, tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    tex = root / "template.tex"
    tex.write_text("A")

    calls = {"compile": 0}

    def fake_compile(cwd, pdf):
        calls["compile"] += 1
        Path(pdf).write_text("pdf")

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="```latex\nA\n```"))])

    monkeypatch.setattr(reflection, "compile_latex", fake_compile)
    monkeypatch.setattr(reflection, "create_client", lambda m: (DummyClient(), m))
    monkeypatch.setattr(reflection, "create_vlm_client", lambda m: (object(), m))
    monkeypatch.setattr(reflection, "perform_imgs_cap_ref_review", lambda *a, **k: {})
    monkeypatch.setattr(reflection, "detect_duplicate_figures", lambda *a, **k: {})
    monkeypatch.setattr(reflection, "get_reflection_page_info", lambda *a, **k: "")
    monkeypatch.setattr(reflection.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))

    reflection.reflect_paper(root, num_rounds=3, model="m", vlm_model="v")
    # compile should be called once in loop and once at end -> 2
    assert calls["compile"] == 2


def test_cli_defaults(monkeypatch, tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    (project / "template.tex").write_text("T")
    ideas = tmp_path / "ideas.json"
    ideas.write_text("{}")

    captured = {}

    def fake_improve(latex_dir, seed_ideas, human_reviews=None, **kwargs):
        captured["kwargs"] = kwargs
        captured["latex_dir"] = Path(latex_dir)
        captured["seed_ideas"] = seed_ideas
        captured["human_reviews"] = human_reviews

    monkeypatch.setattr("ai_scientist.paper_improver.pipeline.improve_paper", fake_improve)
    monkeypatch.setattr("ai_scientist.paper_improver.improve_paper", fake_improve)

    argv = ["prog", str(project), str(ideas)]
    monkeypatch.setattr(__import__("sys"), "argv", argv)
    import runpy

    runpy.run_path("scripts/launch_paper_improver.py", run_name="__main__")

    kw = captured["kwargs"]
    assert kw["strategy"] == "bfs"
    assert kw["max_depth"] == 2
    assert kw["beam_size"] == 3
    assert kw["num_drafts"] == 3
    assert kw["debug_prob"] == 0.5
    assert kw["max_debug_depth"] == 3
    assert kw["model_editor"]
    assert kw["model_review"]
    assert kw["model_vlm"]
    assert kw["orchestrator_model"]
    assert kw["model_citation"]
    assert kw["num_cite_rounds"] == 20
    assert kw["model_reflection"]
    assert kw["num_reflections"] == 3
    assert kw["page_limit"] == 4
