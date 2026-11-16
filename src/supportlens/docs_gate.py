"""Leader-owned Docs-QA gate; Banking keeps its original Gate and fit protocol."""

from dataclasses import dataclass

from .evaluation import Gate, risk_coverage


@dataclass(frozen=True)
class DocsGate(Gate):
    score_name: str | None = None
    target_k: int | None = None
    selection: str = "risk"
    target_coverage: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.target_k is not None and self.target_k < 1:
            raise ValueError("target_k must be positive")
        if self.target_coverage is not None and not 0 < self.target_coverage <= 1:
            raise ValueError("target_coverage outside (0,1]")


def select_coverage_gate(scores: list[float], errors: list[bool], *, method: str,
                         artifact_version: str, split: str, score_name: str,
                         target_k: int = 3, target_coverage: float = 0.7) -> DocsGate:
    """Smallest attainable coverage >= target, accepting complete score ties.

    Labels report risk; they do not choose an optimistic threshold. This does
    not promise a risk bound, answer correctness or general unanswerability.
    """
    if split != "dev":
        raise ValueError("do not fit a threshold on test/held-out")
    if not 0 < target_coverage <= 1 or not score_name or target_k < 1:
        raise ValueError("valid target coverage, score policy and k required")
    curve = risk_coverage(scores, errors)
    i, best = next((i, p) for i, p in enumerate(curve) if p["coverage"] >= target_coverage)
    threshold = best["threshold"]
    if i + 1 < len(curve):
        # Same accepted groups, with margin against batch/single float drift.
        threshold = threshold / 2 + curve[i + 1]["threshold"] / 2
    return DocsGate(method, artifact_version, threshold, split, score_name,
                    target_k, "coverage", target_coverage)
