from ai_scientist.paper_improver.utils import unique_subdir


def test_unique_subdir(tmp_path):
    p1 = unique_subdir(tmp_path, "foo")
    p2 = unique_subdir(tmp_path, "foo")
    assert p1 != p2
    assert p1.parent == tmp_path
    assert not p1.exists()
    assert not p2.exists()
