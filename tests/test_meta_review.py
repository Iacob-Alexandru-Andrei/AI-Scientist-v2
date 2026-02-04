import math
from ai_scientist.paper_improver.meta_review import score_single, meta_score


def test_score_single():
    review = {
        "Originality": 2,
        "Quality": 3,
        "Clarity": 3,
        "Significance": 4,
        "Overall": 8,
    }
    expected = (3*2 + 3*3 + 2*3 + 4*4 + 5*(8/2.5)) / 17
    assert math.isclose(score_single(review), expected)


def test_meta_score_average():
    r1 = {"Originality": 1, "Quality": 1, "Clarity": 1, "Significance": 1, "Overall": 5}
    r2 = {"Originality": 2, "Quality": 2, "Clarity": 2, "Significance": 2, "Overall": 6}
    expected = (score_single(r1) + score_single(r2)) / 2
    assert math.isclose(meta_score([r1, r2]), expected)
