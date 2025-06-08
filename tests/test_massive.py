import pytest
from ai_scientist.paper_improver import meta_review, utils, search


@pytest.mark.parametrize("overall", range(1, 51))
def test_score_single_range(overall):
    review = {
        "Originality": 1,
        "Quality": 1,
        "Clarity": 1,
        "Significance": 1,
        "Overall": overall,
    }
    score = meta_review.score_single(review)
    assert score > 0


@pytest.mark.parametrize("prefix", [f"p{i}" for i in range(25)])
def test_unique_subdir_multiple(tmp_path, prefix):
    paths = {utils.unique_subdir(tmp_path, prefix) for _ in range(4)}
    assert len(paths) == 4
    assert all(not p.exists() and p.parent == tmp_path for p in paths)


@pytest.mark.parametrize("idx", range(25))
def test_searchparams_fields(idx):
    p = search.SearchParams(
        max_depth=idx + 1,
        beam_size=idx + 2,
        num_drafts=idx % 3,
        debug_prob=0.1,
        max_debug_depth=idx + 3,
    )
    assert p.max_depth == idx + 1
    assert p.beam_size == idx + 2
    assert p.num_drafts == idx % 3
    assert p.debug_prob == 0.1
    assert p.max_debug_depth == idx + 3
