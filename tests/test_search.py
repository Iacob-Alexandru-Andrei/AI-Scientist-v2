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

    def patched_evaluate(self):
        fake_compile(self.latex_dir, self.pdf_path)
        self.llm_json = fake_review(str(self.pdf_path))
        self.vlm_json = fake_review(str(self.pdf_path))
        self.score = search.meta_score([self.llm_json, self.vlm_json])
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
