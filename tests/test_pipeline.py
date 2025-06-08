from pathlib import Path
from ai_scientist.paper_improver import pipeline


def test_improve_paper_strategy(monkeypatch, tmp_path):
    called = {}
    root = tmp_path / "paper"
    root.mkdir()
    (root / "template.tex").write_text("test")

    def fake_bfs(*args, **kwargs):
        called['bfs'] = True
        class R:
            latex_dir = Path('bfs')
        return R(), None

    def fake_tree(*args, **kwargs):
        called['tree'] = True
        class R:
            latex_dir = Path('tree')
        return R(), None

    monkeypatch.setattr(pipeline, 'breadth_first_improve', fake_bfs)
    monkeypatch.setattr(pipeline, 'tree_search_improve', fake_tree)

    pipeline.improve_paper(root, 'ideas', strategy='bfs')
    pipeline.improve_paper(root, 'ideas', strategy='tree')

    assert called == {'bfs': True, 'tree': True}


def test_improve_paper_citations(monkeypatch, tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    (root / "template.tex").write_text("tex")

    def fake_bfs(*args, **kwargs):
        class R:
            latex_dir = root
        return R(), None

    captured = {}

    def fake_gather(path, num_cite_rounds=20, small_model="m"):
        captured["args"] = (Path(path), num_cite_rounds, small_model)

    monkeypatch.setattr(pipeline, "breadth_first_improve", fake_bfs)
    monkeypatch.setattr(pipeline, "gather_citations", fake_gather)

    pipeline.improve_paper(
        root,
        "ideas",
        model_citation="cmodel",
        num_cite_rounds=3,
    )

    assert captured["args"] == (root.resolve(), 3, "cmodel")


def test_improve_paper_hyperparams_bfs(monkeypatch, tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "template.tex").write_text("t")

    captured = {}

    def fake_bfs(
        root_dir,
        seed_ideas,
        human_reviews,
        *,
        params,
        model_editor,
        model_review,
        model_vlm,
        orchestrator_model,
        **kwargs,
    ):
        captured["root_dir"] = Path(root_dir)
        captured["params"] = params
        captured["editor"] = model_editor
        captured["review"] = model_review
        captured["vlm"] = model_vlm
        captured["orch"] = orchestrator_model
        class R:
            latex_dir = root_dir
        return R(), None

    monkeypatch.setattr(pipeline, "breadth_first_improve", fake_bfs)
    monkeypatch.setattr(pipeline, "gather_citations", lambda *a, **k: None)

    pipeline.improve_paper(
        root,
        "ideas",
        strategy="bfs",
        model_editor="e",
        model_review="r",
        model_vlm="v",
        orchestrator_model="o",
        max_depth=5,
        beam_size=2,
        num_drafts=1,
        debug_prob=0.1,
        max_debug_depth=4,
        num_cite_rounds=0,
    )

    p = captured["params"]
    assert p.max_depth == 5
    assert p.beam_size == 2
    assert p.num_drafts == 1
    assert p.debug_prob == 0.1
    assert p.max_debug_depth == 4
    assert captured["editor"] == "e"
    assert captured["review"] == "r"
    assert captured["vlm"] == "v"
    assert captured["orch"] == "o"


def test_improve_paper_hyperparams_tree(monkeypatch, tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    (root / "template.tex").write_text("t")

    captured = {}

    def fake_tree(
        root_dir,
        seed_ideas,
        human_reviews,
        *,
        params,
        model_editor,
        model_review,
        model_vlm,
        orchestrator_model,
        **kwargs,
    ):
        captured["root_dir"] = Path(root_dir)
        captured["params"] = params
        captured["editor"] = model_editor
        captured["review"] = model_review
        captured["vlm"] = model_vlm
        captured["orch"] = orchestrator_model
        class R:
            latex_dir = root_dir
        return R(), None

    monkeypatch.setattr(pipeline, "tree_search_improve", fake_tree)
    monkeypatch.setattr(pipeline, "gather_citations", lambda *a, **k: None)

    pipeline.improve_paper(
        root,
        "ideas",
        strategy="tree",
        model_editor="e2",
        model_review="r2",
        model_vlm="v2",
        orchestrator_model="o2",
        max_depth=4,
        beam_size=5,
        num_drafts=2,
        debug_prob=0.2,
        max_debug_depth=5,
        num_cite_rounds=0,
    )

    p = captured["params"]
    assert p.max_depth == 4
    assert p.beam_size == 5
    assert p.num_drafts == 2
    assert p.debug_prob == 0.2
    assert p.max_debug_depth == 5
    assert captured["editor"] == "e2"
    assert captured["review"] == "r2"
    assert captured["vlm"] == "v2"
    assert captured["orch"] == "o2"


def test_citation_defaults(monkeypatch, tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    (root / "template.tex").write_text("t")

    def fake_bfs(*args, **kwargs):
        class R:
            latex_dir = root
        return R(), None

    captured = {}

    def fake_gather(path, num_cite_rounds=20, small_model="m"):
        captured["args"] = (Path(path), num_cite_rounds, small_model)

    monkeypatch.setattr(pipeline, "breadth_first_improve", fake_bfs)
    monkeypatch.setattr(pipeline, "gather_citations", fake_gather)

    pipeline.improve_paper(root, "ideas", num_cite_rounds=2)

    assert captured["args"] == (root.resolve(), 2, pipeline.CITATION_MODEL)

