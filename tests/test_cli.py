import sys
import runpy
from pathlib import Path

def test_cli_parsing(monkeypatch, tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    (project / "template.tex").write_text("t")
    ideas = tmp_path / "ideas.json"
    ideas.write_text("{}")
    reviews = tmp_path / "rev.txt"
    reviews.write_text("r")

    captured = {}

    def fake_improve(latex_dir, seed_ideas, human_reviews=None, **kwargs):
        captured["latex_dir"] = Path(latex_dir)
        captured["seed_ideas"] = seed_ideas
        captured["human_reviews"] = human_reviews
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "ai_scientist.paper_improver.pipeline.improve_paper", fake_improve
    )
    # __init__ re-exports improve_paper; patch it too so the CLI uses our stub
    monkeypatch.setattr(
        "ai_scientist.paper_improver.improve_paper", fake_improve
    )

    argv = [
        "prog",
        str(project),
        str(ideas),
        "--human-reviews",
        str(reviews),
        "--max-depth",
        "1",
        "--beam-size",
        "2",
        "--num-drafts",
        "3",
        "--debug-prob",
        "0.3",
        "--max-debug-depth",
        "4",
        "--strategy",
        "tree",
        "--model-editor",
        "E",
        "--model-review",
        "R",
        "--model-vlm",
        "V",
        "--model-orchestrator",
        "O",
        "--model-citation",
        "C",
        "--num-cite-rounds",
        "5",
        "--model-reflection",
        "M",
        "--num-reflections",
        "2",
        "--page-limit",
        "7",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path("scripts/launch_paper_improver.py", run_name="__main__")

    assert captured["latex_dir"] == project
    assert captured["seed_ideas"] == ideas.read_text()
    assert captured["human_reviews"] == reviews.read_text()
    kw = captured["kwargs"]
    assert kw["max_depth"] == 1
    assert kw["beam_size"] == 2
    assert kw["num_drafts"] == 3
    assert kw["debug_prob"] == 0.3
    assert kw["max_debug_depth"] == 4
    assert kw["strategy"] == "tree"
    assert kw["model_editor"] == "E"
    assert kw["model_review"] == "R"
    assert kw["model_vlm"] == "V"
    assert kw["orchestrator_model"] == "O"
    assert kw["model_citation"] == "C"
    assert kw["num_cite_rounds"] == 5
    assert kw["model_reflection"] == "M"
    assert kw["num_reflections"] == 2
    assert kw["page_limit"] == 7
