from pathlib import Path
from ai_scientist.paper_improver import search


def setup_dummy_env(tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    (root / "template.tex").write_text("Start")
    return root


def fake_compile(cwd, pdf_file):
    src = Path(cwd) / "template.tex"
    Path(pdf_file).write_text(src.read_text())


def fake_propose_edit(path, seed_ideas, human_reviews, model=None):
    return path.read_text() + "X"


def fake_review(pdf_path):
    text = Path(pdf_path).read_text()
    overall = min(10, len(text))
    return {
        "Originality": 1,
        "Quality": 1,
        "Clarity": 1,
        "Significance": 1,
        "Overall": overall,
    }


class DummyJournal(search.Journal):
    # override best_node to avoid LLM call
    def best_node(self, orchestrator_model=search.ORCHESTRATOR_MODEL):
        return max(self.nodes, key=lambda n: n.score or 0)


def run_search(impl, tmp_path, monkeypatch):
    root = setup_dummy_env(tmp_path)

    counter = {"i": 0}

    def patched_evaluate(self):
        counter["i"] += 1
        self.score = float(counter["i"])
        return self.score

    monkeypatch.setattr(search.PaperNode, "evaluate", patched_evaluate)
    monkeypatch.setattr(
        search.PaperNode,
        "compile",
        lambda self: fake_compile(self.latex_dir, self.pdf_path),
    )
    monkeypatch.setattr(search, "propose_edit", fake_propose_edit)
    monkeypatch.setattr(search, "llm_review", fake_review)
    monkeypatch.setattr(search, "vlm_review", fake_review)
    monkeypatch.setattr(search, "Journal", DummyJournal)
    params = search.SearchParams(max_depth=2, beam_size=1, num_drafts=0)
    best, journal = impl(root, "ideas", None, params=params)
    for n in journal.nodes:
        print("NODE", n.depth, n.score)
        assert n.score is not None
    return best, journal


def test_bfs_and_tree_search(monkeypatch, tmp_path):
    best_bfs, _ = run_search(search.breadth_first_improve, tmp_path, monkeypatch)
    assert best_bfs.depth == 2

    tmp2 = tmp_path / "other"
    tmp2.mkdir()
    best_tree, _ = run_search(search.tree_search_improve, tmp2, monkeypatch)
    assert best_tree.depth == 2


def test_search_params_usage(monkeypatch, tmp_path):
    root = setup_dummy_env(tmp_path)
    propose_calls = []

    def patched_propose(path, seed_ideas, human_reviews, model=None):
        propose_calls.append(path)
        return path.read_text() + "X"

    counter = {"i": 0}

    def patched_evaluate(self):
        counter["i"] += 1
        self.score = float(counter["i"])
        return self.score

    monkeypatch.setattr(search, "propose_edit", patched_propose)
    monkeypatch.setattr(search.PaperNode, "evaluate", patched_evaluate)
    monkeypatch.setattr(search.PaperNode, "compile", lambda self: fake_compile(self.latex_dir, self.pdf_path))
    monkeypatch.setattr(search, "llm_review", fake_review)
    monkeypatch.setattr(search, "vlm_review", fake_review)
    monkeypatch.setattr(search, "Journal", DummyJournal)

    params = search.SearchParams(max_depth=1, beam_size=2, num_drafts=3)
    best, journal = search.breadth_first_improve(root, "ideas", None, params=params)

    assert len(propose_calls) == 5  # 3 drafts + 2 expansions
    assert max(n.depth for n in journal.nodes) == 1
    assert len(journal.nodes) == 6

    tmp2 = tmp_path / "tree"
    tmp2.mkdir()
    propose_calls.clear()
    root2 = setup_dummy_env(tmp2)
    best2, journal2 = search.tree_search_improve(root2, "ideas", None, params=params)

    assert len(propose_calls) == 5
    assert max(n.depth for n in journal2.nodes) == 1
    assert len(journal2.nodes) == 6


def test_debug_retry(monkeypatch, tmp_path):
    root = setup_dummy_env(tmp_path)
    calls = []

    def failing_eval(self):
        calls.append("eval")
        if len(calls) == 1:
            self.score = 0
            raise RuntimeError("fail")
        self.score = 1
        return 1

    monkeypatch.setattr(search.PaperNode, "evaluate", failing_eval)
    monkeypatch.setattr(search, "propose_edit", lambda *a, **k: "X")
    monkeypatch.setattr(search.PaperNode, "compile", lambda self: None)
    monkeypatch.setattr(search, "llm_review", lambda *a, **k: {})
    monkeypatch.setattr(search, "vlm_review", lambda *a, **k: {})
    monkeypatch.setattr(search, "Journal", DummyJournal)
    monkeypatch.setattr(search.random, "random", lambda: 0.0)

    params = search.SearchParams(max_depth=1, beam_size=0, num_drafts=0, debug_prob=1.0, max_debug_depth=1)
    best, journal = search.breadth_first_improve(root, "ideas", None, params=params)

    assert len(calls) == 2
    assert best.debug_depth == 1
    assert len(journal.nodes) == 1


def test_orchestrator_model(monkeypatch):
    j = search.Journal()
    n1 = search.PaperNode(Path("a"))
    n2 = search.PaperNode(Path("b"))
    n1.score = 1
    n2.score = 2
    j.append(n1)
    j.append(n2)

    captured = {}

    def fake_query(**kwargs):
        captured["model"] = kwargs.get("model")
        return {"selected_id": n2.id, "reasoning": ""}

    monkeypatch.setattr(search, "query", fake_query)
    chosen = j.best_node("orch")

    assert captured["model"] == "orch"
    assert chosen is n2
