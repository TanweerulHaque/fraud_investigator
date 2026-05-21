"""Build structured explainability artifact from investigation state."""

from __future__ import annotations

from typing import Any

from schemas.case import BehavioralSummary, CriticOutput, ExplanationOutput, RetrievedCase
from schemas.case import ReviewerDecision
from schemas.output import ExplainabilityArtifact


def build_explainability_artifact(
    case_id: str,
    decision: ReviewerDecision,
    confidence: float,
    decision_drivers: list[str],
    behavioral: BehavioralSummary | None,
    retrieved: list[RetrievedCase],
    critic: CriticOutput | None,
    explanation: ExplanationOutput | None,
    agent_trace: list[dict[str, Any]],
) -> ExplainabilityArtifact:
    signals = list(decision_drivers)
    if behavioral:
        signals.extend([f"{i.signal}: {i.severity}" for i in behavioral.risk_indicators[:5]])

    conf_rationale = _confidence_rationale(confidence, decision, critic)

    return ExplainabilityArtifact(
        decision=decision,
        confidence=confidence,
        confidence_rationale=conf_rationale,
        top_evidence_signals=signals[:10],
        similar_cases=retrieved,
        counterfactuals=critic.counterfactuals if critic else [],
        uncertainty_notes=critic.uncertainty_notes if critic else [],
        contradictory_evidence=critic.contradictory_evidence if critic else [],
        analyst_rationale=explanation.analyst_rationale if explanation else "",
        top_reasons=explanation.top_reasons if explanation else decision_drivers[:5],
        risk_narrative=explanation.risk_narrative if explanation else "",
        explanation_trace=agent_trace,
    )


def _confidence_rationale(
    confidence: float, decision: ReviewerDecision, critic: CriticOutput | None
) -> str:
    parts = []
    if confidence >= 0.75:
        parts.append("High confidence: multiple independent risk signals align.")
    elif confidence >= 0.55:
        parts.append("Moderate confidence: mixed signals; judgment weighs DS and behavior.")
    else:
        parts.append("Lower confidence: conflicting signals; binary judgment still required.")

    if critic and critic.contradictory_evidence:
        parts.append(
            f"Critic noted {len(critic.contradictory_evidence)} contradictory item(s), tempering certainty."
        )
    return " ".join(parts)
