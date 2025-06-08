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
