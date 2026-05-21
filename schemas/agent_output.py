"""Dual agent output: raw (inter-agent) and structured (downstream)."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class AgentDualOutput(BaseModel, Generic[T]):
    """Every agent publishes raw reasoning plus typed structured output."""

    agent_name: str
    raw: dict[str, Any] = Field(default_factory=dict)
    structured: dict[str, Any] = Field(default_factory=dict)

    def get_structured(self, model: type[T]) -> T:
        return model.model_validate(self.structured)


class IntakeStructured(BaseModel):
    case_id: str
    quality_flags: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    intake_notes: str = ""


class EmbeddingStructured(BaseModel):
    case_id: str
    case_summary: str
    feature_dim: int
    summary_embedding_dim: int
    combined_dim: int


class BehavioralStructured(BaseModel):
    summary: str
    risk_indicators: list[dict[str, Any]] = Field(default_factory=list)
    stability_notes: str = ""
    stable_signals: list[str] = Field(default_factory=list)
    drifting_signals: list[str] = Field(default_factory=list)


class RiskSynthesisStructured(BaseModel):
    decision: str  # fraud | not_fraud
    confidence: float
    decision_drivers: list[str] = Field(default_factory=list)
    reasoning: str = ""


class CriticStructured(BaseModel):
    contradictory_evidence: list[str] = Field(default_factory=list)
    counterfactuals: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ExplanationStructured(BaseModel):
    analyst_rationale: str
    top_reasons: list[str] = Field(default_factory=list)
    risk_narrative: str = ""


class RetrievalStructured(BaseModel):
    retrieved_count: int
    matched_patterns: list[str] = Field(default_factory=list)
    neighbor_fraud_votes: int = 0
    neighbor_not_fraud_votes: int = 0


class MemoryUpdateStructured(BaseModel):
    case_id: str
    stored: bool
    chroma_id: str = ""
