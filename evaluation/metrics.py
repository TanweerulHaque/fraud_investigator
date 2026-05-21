"""Evaluation metrics for fraud twin PoC."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float] | None = None,
) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class = {
        lbl: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for lbl, p, r, f, s in zip(labels, prec, rec, f1, support)
    }

    # Fraud-specific FPR/FNR (fraud as positive)
    fraud_idx = labels.index("fraud") if "fraud" in labels else None
    fpr = fnr = None
    if fraud_idx is not None:
        tp = fp = tn = fn = 0
        for t, p in zip(y_true, y_pred):
            if t == "fraud" and p == "fraud":
                tp += 1
            elif t != "fraud" and p == "fraud":
                fp += 1
            elif t == "fraud" and p != "fraud":
                fn += 1
            else:
                tn += 1
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0

    calibration = _simple_calibration(y_true, y_pred, confidences or [])

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "false_positive_rate_fraud": fpr,
        "false_negative_rate_fraud": fnr,
        "calibration": calibration,
    }


def _simple_calibration(
    y_true: list[str], y_pred: list[str], confidences: list[float]
) -> dict[str, Any]:
    if not confidences:
        return {"note": "no confidences"}
    correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
    bins = defaultdict(list)
    for c, ok in zip(confidences, correct):
        b = min(9, int(c * 10))
        bins[b].append(ok)
    bin_acc = {str(k): float(np.mean(v)) for k, v in bins.items()}
    return {"bin_accuracy": bin_acc, "mean_confidence": float(np.mean(confidences))}


def explanation_quality_proxy(result: dict[str, Any]) -> float:
    """0-1 proxy: required explainability fields populated."""
    exp = result.get("explainability") or result
    checks = [
        bool(exp.get("analyst_rationale") or exp.get("decision")),
        len(exp.get("top_evidence_signals") or exp.get("top_reasons") or []) >= 1,
        "confidence" in exp or result.get("confidence") is not None,
    ]
    return sum(checks) / len(checks)


def retrieval_relevance(
    pred_label: str, retrieved: list[dict[str, Any]]
) -> float:
    if not retrieved:
        return 0.0
    matches = sum(1 for r in retrieved if r.get("decision") == pred_label)
    return matches / len(retrieved)
