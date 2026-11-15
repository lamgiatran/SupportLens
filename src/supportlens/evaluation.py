"""Shared, tie-aware risk/coverage for banking and Docs-QA retrieval failure.

Docs error = no annotated relevant passage in top-5. This is a retrieval proxy,
not an answer-factuality or general unanswerability label. Use method-specific
scores and thresholds; BM25, cosine and RRF scores are not interchangeable.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Gate:
    method: str
    artifact_version: str
    threshold: float | None  # None means deny-all after fitting, not unconfigured.
    fitted_split: str

    def __post_init__(self):
        if self.fitted_split != "dev":
            raise ValueError("thresholds must be selected on dev")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("non-finite threshold")

    def accepts(self, score: float) -> bool:
        return self.threshold is not None and math.isfinite(score) and score >= self.threshold


def risk_coverage(scores: list[float], errors: list[bool]) -> list[dict]:
    if len(scores) != len(errors) or not scores:
        raise ValueError("non-empty aligned scores/errors required")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite")
    ranked = sorted(zip(scores, errors), key=lambda row: -row[0])
    points = [{"threshold": None, "accepted": 0, "coverage": 0.0, "risk": None}]
    wrong = 0
    for i, (score, error) in enumerate(ranked, 1):
        wrong += int(error)
        # All tied scores must be accepted together by a real scalar threshold.
        if i == len(ranked) or score != ranked[i][0]:
            points.append({"threshold": score, "accepted": i,
                           "coverage": i / len(ranked), "risk": wrong / i})
    return points


def select_gate(scores: list[float], errors: list[bool], *, method: str,
                artifact_version: str, split: str, target_risk: float = 0.1) -> Gate:
    if split != "dev":
        raise ValueError("do not fit a threshold on test/held-out")
    if not 0 <= target_risk <= 1:
        raise ValueError("target_risk outside [0,1]")
    eligible = [p for p in risk_coverage(scores, errors)
                if p["risk"] is not None and p["risk"] <= target_risk]
    best = max(eligible, key=lambda p: p["coverage"]) if eligible else None
    return Gate(method, artifact_version, best["threshold"] if best else None, split)
