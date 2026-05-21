"""Mock DS model score context lookup."""

from __future__ import annotations

from schemas.case import ModelScores


def enrich_model_score_context(scores: ModelScores) -> dict[str, str]:
    """Return interpretable bands for model scores (no external API)."""
    def band(v: float) -> str:
        if v >= 0.8:
            return "critical"
        if v >= 0.6:
            return "elevated"
        if v >= 0.4:
            return "moderate"
        return "low"

    return {
        "fraud_score_band": band(scores.fraud_score),
        "aml_score_band": band(scores.aml_score),
        "ato_score_band": band(scores.account_takeover_score),
        "composite_note": (
            "DS ensemble elevated" if scores.fraud_score >= 0.65 or scores.account_takeover_score >= 0.6
            else "DS ensemble within normal band"
        ),
    }
