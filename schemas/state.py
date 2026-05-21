"""LangGraph shared state."""

from __future__ import annotations

from typing import Any, TypedDict

from schemas.agent_output import AgentDualOutput
from schemas.case import (
    BehavioralSummary,
    CriticOutput,
    DecisionDraft,
    ExplanationOutput,
    NormalizedCase,
    RetrievedCase,
    ReviewerDecision,
)


class InvestigatorState(TypedDict, total=False):
    """Mutable state passed through the LangGraph pipeline."""

    raw_case: dict[str, Any]
    normalized_case: NormalizedCase | None
    missing_fields: list[str]
    quality_flags: list[str]

    case_summary: str | None
    case_embedding: list[float] | None

    retrieved_cases: list[RetrievedCase]
    matched_patterns: list[str]
    prior_reasoning_snippets: list[str]

    behavioral_summary: BehavioralSummary | None

    decision_draft: DecisionDraft | None

    critic_output: CriticOutput | None
    explanation_output: ExplanationOutput | None

    decision: ReviewerDecision | None
    confidence: float | None
    decision_drivers: list[str]

    agent_outputs: dict[str, dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    memory_updated: bool
    error: str | None


def pack_agent_output(output: AgentDualOutput) -> dict[str, Any]:
    return output.model_dump()
