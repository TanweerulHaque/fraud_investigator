"""Final investigation output and explainability artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from schemas.case import (
    BehavioralSummary,
    CriticOutput,
    ExplanationOutput,
    RetrievedCase,
    ReviewerDecision,
)


class ExplainabilityArtifact(BaseModel):
    """Structured explainability — not free-form only."""

    decision: ReviewerDecision
    confidence: float
    confidence_rationale: str
    top_evidence_signals: list[str]
    similar_cases: list[RetrievedCase]
    counterfactuals: list[str]
    uncertainty_notes: list[str]
    contradictory_evidence: list[str]
    analyst_rationale: str
    top_reasons: list[str]
    risk_narrative: str
    explanation_trace: list[dict[str, Any]] = Field(default_factory=list)

    def to_analyst_sections(self) -> dict[str, Any]:
        return {
            "what_drove_the_decision": self.top_evidence_signals,
            "similar_past_cases": [
                {"case_id": c.case_id, "similarity": c.similarity, "decision": c.decision}
                for c in self.similar_cases
            ],
            "what_would_change_outcome": self.counterfactuals,
            "why_confidence": self.confidence_rationale,
            "uncertainty": self.uncertainty_notes,
            "rationale": self.analyst_rationale,
        }


class InvestigationResult(BaseModel):
    case_id: str
    decision: ReviewerDecision
    confidence: float
    decision_drivers: list[str]
    behavioral_summary: BehavioralSummary | None = None
    retrieved_cases: list[RetrievedCase] = Field(default_factory=list)
    critic: CriticOutput | None = None
    explanation: ExplanationOutput | None = None
    explainability: ExplainabilityArtifact
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    agent_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
