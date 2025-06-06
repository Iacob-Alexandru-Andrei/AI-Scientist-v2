"""Combine multiple review JSONs into a single numerical score."""
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


def score_single(review_json: Dict[str, Any], weights: Dict[str, int] | None = None) -> float:
    """Weighted average of key fields (1-10 scaled to 1-4 where needed)."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    total, denom = 0.0, 0.0
    for k, w in weights.items():
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
    """Average score over a set of reviews."""
    return mean(score_single(r) for r in reviews) if reviews else 0.0
