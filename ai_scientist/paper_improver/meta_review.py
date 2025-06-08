"""Utility functions for aggregating review scores.

``score_single`` computes a weighted average over the numeric fields present
in a single review JSON object.  ``meta_score`` then averages those scores
across multiple reviewers.  These helpers are intentionally simple; they only
look at a few high-level categories and normalise the "Overall" field onto the
same 1–4 scale as the others.  The resulting float is used by the search code
to rank paper versions.
"""

from __future__ import annotations
from statistics import mean
from typing import Sequence, Dict, Any

DEFAULT_WEIGHTS = {
    "Originality": 3,
    "Quality": 3,
    "Clarity": 2,
    "Significance": 4,
    "Overall": 5,  # emphasise overall judgement
}


def score_single(
    review_json: Dict[str, Any], weights: Dict[str, int] | None = None
) -> float:
    """Weighted average of key fields (1-10 scaled to 1-4 where needed)."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    total, denom = 0.0, 0.0  # running numerator/denominator for the average
    for k, w in weights.items():
        if review_json is None:
            continue
        val = review_json.get(k)
        if val is None:
            continue
        # normalise different scales
        if k == "Overall":
            val = val / 2.5  # map 1-10 → ~0-4
        total += w * float(val)
        denom += w
    return total / denom if denom else 0.0


def meta_score(reviews: Sequence[Dict[str, Any]]) -> float:
    """Average ``score_single`` over a set of reviews."""
    return mean(score_single(r) for r in reviews) if reviews else 0.0
