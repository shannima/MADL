"""Dependency-light classification and binary localization metrics."""

from __future__ import annotations

from typing import Iterable, Sequence

LABELS = ("real", "synthetic", "tampered")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    targets: Sequence[str],
    predictions: Sequence[str],
    labels: Iterable[str] = LABELS,
) -> dict:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must have the same non-zero length")
    labels = tuple(str(label) for label in labels)
    if any(value not in labels for value in (*targets, *predictions)):
        raise ValueError("classification values fall outside the configured label set")
    per_class = {}
    f1_values = []
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(targets, predictions))
        false_positive = sum(t != label and p == label for t, p in zip(targets, predictions))
        false_negative = sum(t == label and p != label for t, p in zip(targets, predictions))
        true_negative = len(targets) - true_positive - false_positive - false_negative
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(target == label for target in targets),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        }
        f1_values.append(f1)
    return {
        "accuracy": sum(target == prediction for target, prediction in zip(targets, predictions))
        / len(targets),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
    }


def localization_metrics(prediction, target) -> dict[str, float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - NumPy is a base dependency
        raise RuntimeError("NumPy is required for localization metrics") from exc
    prediction_array = np.asarray(prediction) > 0
    target_array = np.asarray(target) > 0
    if prediction_array.shape != target_array.shape or prediction_array.size == 0:
        raise ValueError("prediction and target masks must have the same non-empty shape")
    true_positive = int(np.logical_and(prediction_array, target_array).sum())
    false_positive = int(np.logical_and(prediction_array, ~target_array).sum())
    false_negative = int(np.logical_and(~prediction_array, target_array).sum())
    union = true_positive + false_positive + false_negative
    iou = _safe_ratio(true_positive, union)
    f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {"iou": iou, "f1": f1}
