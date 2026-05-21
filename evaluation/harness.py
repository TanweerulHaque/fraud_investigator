"""Evaluation harness over labeled mock cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.graph import run_investigation, seed_memory_from_file
from evaluation.metrics import (
    compute_metrics,
    explanation_quality_proxy,
    retrieval_relevance,
)
from tools.case_loader import load_cases_from_json
from tools.validator import validate_case_dict
from utils.config import get_config, resolve_path
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class EvaluationHarness:
    def __init__(self, eval_path: str | None = None) -> None:
        cfg = get_config()
        self.eval_path = resolve_path(eval_path or cfg["paths"]["eval_cases"])

    def load_eval_cases(self) -> list[dict[str, Any]]:
        with open(self.eval_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("cases", [])

    def run(self, verbose: bool = False, reseed: bool = True) -> dict[str, Any]:
        setup_logging()
        if reseed:
            seed_memory_from_file()

        cases = self.load_eval_cases()
        y_true: list[str] = []
        y_pred: list[str] = []
        confidences: list[float] = []
        expl_scores: list[float] = []
        retrieval_scores: list[float] = []
        consistency_checks: list[bool] = []
        per_case: list[dict[str, Any]] = []

        for raw in cases:
            fc, err, _ = validate_case_dict(raw)
            if fc is None or not fc.reviewer_label:
                logger.warning("Skipping case without label: %s", err)
                continue

            label = fc.reviewer_label.value
            result = run_investigation(raw)
            pred = result.decision.value

            y_true.append(label)
            y_pred.append(pred)
            confidences.append(result.confidence)

            res_dict = result.model_dump()
            expl_scores.append(explanation_quality_proxy(res_dict))
            retrieval_scores.append(
                retrieval_relevance(
                    label,
                    [r.model_dump() for r in result.retrieved_cases],
                )
            )

            # Consistency: second run same decision
            result2 = run_investigation(raw)
            consistency_checks.append(result2.decision == result.decision)

            row = {
                "case_id": result.case_id,
                "label": label,
                "prediction": pred,
                "match": label == pred,
                "confidence": result.confidence,
            }
            per_case.append(row)
            if verbose:
                logger.info("%s label=%s pred=%s conf=%.2f", result.case_id, label, pred, result.confidence)

        metrics = compute_metrics(y_true, y_pred, confidences)
        metrics["explanation_quality_proxy_mean"] = float(sum(expl_scores) / len(expl_scores)) if expl_scores else 0
        metrics["retrieval_relevance_mean"] = float(sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else 0
        metrics["consistency_rate"] = float(sum(consistency_checks) / len(consistency_checks)) if consistency_checks else 0
        metrics["cases_evaluated"] = len(y_true)
        metrics["per_case"] = per_case

        return metrics


def run_evaluation(verbose: bool = False) -> dict[str, Any]:
    return EvaluationHarness().run(verbose=verbose)
