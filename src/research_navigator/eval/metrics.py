"""Pure, dependency-free metric functions.

Everything here is a total function over plain values: no I/O, no LLM, no numpy. That
keeps the numbers deterministic and trivially unit-testable, and it is why the retrieval
/ routing / refusal maths can be verified without any of the live stack.

Retrieval precision/recall are computed at the **document level**: the ranked chunk list
is de-duplicated to its ordered unique ``doc_id``s before scoring against the expected
sources, because the golden labels are documents, not chunks."""

from __future__ import annotations

from collections.abc import Sequence


def dedupe_preserving_order(doc_ids: Sequence[str]) -> list[str]:
    """Collapse a ranked chunk-doc list to ordered unique documents."""
    seen: set[str] = set()
    out: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def precision_at_k(retrieved_doc_ids: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the top-k *retrieved documents* that are relevant.

    Denominator is the number of documents actually returned in the top-k (which may be
    < k for a small corpus), not k itself — otherwise a system is unfairly penalised for
    the corpus being small. Returns 0.0 when nothing is retrieved.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top = dedupe_preserving_order(retrieved_doc_ids)[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if d in relevant)
    return hits / len(top)


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the relevant documents recovered within the top-k documents."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    top = set(dedupe_preserving_order(retrieved_doc_ids)[:k])
    hits = len(top & relevant)
    return hits / len(relevant)


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method), no numpy dependency.

    ``pct`` is in (0, 100). Returns 0.0 for an empty input.
    """
    if not 0 < pct < 100:
        raise ValueError("pct must be in (0, 100)")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(ordered):
        return ordered[-1]
    return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])


def accuracy(pairs: Sequence[tuple[str, str]]) -> float:
    """Overall accuracy of (predicted, expected) label pairs."""
    if not pairs:
        return 0.0
    return sum(1 for pred, exp in pairs if pred == exp) / len(pairs)


def per_label_accuracy(pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    """Accuracy grouped by the *expected* label (per-route routing accuracy)."""
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for pred, exp in pairs:
        totals[exp] = totals.get(exp, 0) + 1
        if pred == exp:
            hits[exp] = hits.get(exp, 0) + 1
    return {label: hits.get(label, 0) / n for label, n in totals.items()}


def confusion_counts(pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """expected -> predicted -> count. Handy for the report's mis-route breakdown."""
    matrix: dict[str, dict[str, int]] = {}
    for pred, exp in pairs:
        row = matrix.setdefault(exp, {})
        row[pred] = row.get(pred, 0) + 1
    return matrix


def binary_rates(actual: Sequence[bool], expected: Sequence[bool]) -> dict[str, float]:
    """Refusal-correctness rates treating 'refused' as the positive class.

    Returns accuracy plus precision/recall/f1 of refusal, so the report can show both
    "does it refuse when it should" (recall) and "does it over-refuse" (precision).
    """
    if len(actual) != len(expected):
        raise ValueError("actual and expected must be the same length")
    if not actual:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0.0}
    tp = sum(1 for a, e in zip(actual, expected, strict=True) if a and e)
    fp = sum(1 for a, e in zip(actual, expected, strict=True) if a and not e)
    fn = sum(1 for a, e in zip(actual, expected, strict=True) if not a and e)
    correct = sum(1 for a, e in zip(actual, expected, strict=True) if a == e)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": correct / len(actual),
        "precision": prec,
        "recall": rec,
        "f1": f1(prec, rec),
        "n": float(len(actual)),
    }
