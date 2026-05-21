"""Pydantic models for fraud case input and normalized representation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ReviewerDecision(str, Enum):
    FRAUD = "fraud"
    NOT_FRAUD = "not_fraud"


class UserProfile(BaseModel):
    user_id: str
    account_age_days: int = Field(ge=0)
    kyc_status: str = "verified"
    prior_disputes: int = 0


class InstalledApp(BaseModel):
    package_name: str
    risk_category: str = "unknown"  # finance, gambling, vpn, utility, unknown
    install_days_ago: int = 0


class Transaction(BaseModel):
    txn_id: str
    amount: float
    currency: str = "INR"
    merchant_category: str
    channel: str  # upi, card, neft, atm
    timestamp: datetime
    is_cash_out: bool = False


class AppUsagePattern(BaseModel):
    sessions_per_day: float = 0.0
    night_usage_ratio: float = 0.0
    unique_locations_7d: int = 0


class LocationPattern(BaseModel):
    home_country: str = "IN"
    countries_7d: list[str] = Field(default_factory=list)
    max_geo_velocity_kmph: float = 0.0


class DeviceAttributes(BaseModel):
    device_id: str
    os: str
    is_emulator: bool = False
    is_rooted: bool = False
    device_changes_30d: int = 0


class ModelScores(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0)
    aml_score: float = Field(ge=0.0, le=1.0, default=0.0)
    account_takeover_score: float = Field(ge=0.0, le=1.0, default=0.0)


class PriorReview(BaseModel):
    case_id: str
    reviewer_decision: ReviewerDecision
    reviewer_notes: str = ""
    reviewed_at: datetime | None = None


class FraudCase(BaseModel):
    """Raw incoming case payload."""

    case_id: str
    user_profile: UserProfile
    installed_apps: list[InstalledApp] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)
    app_usage: AppUsagePattern = Field(default_factory=AppUsagePattern)
    location_patterns: LocationPattern = Field(default_factory=LocationPattern)
    device: DeviceAttributes
    model_scores: ModelScores
    prior_reviews: list[PriorReview] = Field(default_factory=list)
    reviewer_label: ReviewerDecision | None = None  # ground truth for eval only
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("transactions", mode="before")
    @classmethod
    def parse_txn_timestamps(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        out = []
        for t in v:
            if isinstance(t, dict) and isinstance(t.get("timestamp"), str):
                t = {**t, "timestamp": datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))}
            out.append(t)
        return out


class NormalizedCase(BaseModel):
    """Validated, normalized case ready for agents."""

    case_id: str
    user_profile: UserProfile
    installed_apps: list[InstalledApp]
    transactions: list[Transaction]
    app_usage: AppUsagePattern
    location_patterns: LocationPattern
    device: DeviceAttributes
    model_scores: ModelScores
    prior_reviews: list[PriorReview]
    reviewer_label: ReviewerDecision | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Derived features (intake enrichment)
    total_txn_amount_7d: float = 0.0
    cash_out_ratio: float = 0.0
    high_risk_app_count: int = 0
    primary_risk_tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_fraud_case(cls, case: FraudCase) -> NormalizedCase:
        total = sum(t.amount for t in case.transactions)
        cash = sum(1 for t in case.transactions if t.is_cash_out)
        ratio = cash / len(case.transactions) if case.transactions else 0.0
        hr_apps = sum(
            1 for a in case.installed_apps if a.risk_category in ("gambling", "vpn", "clone")
        )
        tags: list[str] = []
        if case.device.is_emulator or case.device.is_rooted:
            tags.append("device_integrity")
        if case.location_patterns.max_geo_velocity_kmph > 800:
            tags.append("geo_velocity")
        if ratio > 0.4:
            tags.append("cash_out")
        if case.model_scores.fraud_score > 0.7:
            tags.append("high_ds_score")
        if case.user_profile.account_age_days < 14:
            tags.append("account_freshness")

        return cls(
            case_id=case.case_id,
            user_profile=case.user_profile,
            installed_apps=case.installed_apps,
            transactions=case.transactions,
            app_usage=case.app_usage,
            location_patterns=case.location_patterns,
            device=case.device,
            model_scores=case.model_scores,
            prior_reviews=case.prior_reviews,
            reviewer_label=case.reviewer_label,
            metadata=case.metadata,
            total_txn_amount_7d=total,
            cash_out_ratio=ratio,
            high_risk_app_count=hr_apps,
            primary_risk_tags=tags,
        )


class RetrievedCase(BaseModel):
    case_id: str
    similarity: float
    decision: str | None = None
    primary_risk_tags: list[str] = Field(default_factory=list)
    reviewer_notes_snippet: str = ""
    reward_representation: dict[str, Any] | None = None


class RiskIndicator(BaseModel):
    signal: str
    severity: str  # low, medium, high
    detail: str
    weight: float = 0.0


class BehavioralSummary(BaseModel):
    summary: str
    risk_indicators: list[RiskIndicator]
    stability_notes: str
    stable_signals: list[str] = Field(default_factory=list)
    drifting_signals: list[str] = Field(default_factory=list)


class DecisionDraft(BaseModel):
    decision: ReviewerDecision
    confidence: float
    decision_drivers: list[str]


class CriticOutput(BaseModel):
    contradictory_evidence: list[str]
    counterfactuals: list[str]
    uncertainty_notes: list[str]


class ExplanationOutput(BaseModel):
    analyst_rationale: str
    top_reasons: list[str]
    risk_narrative: str


class RewardRepresentation(BaseModel):
    """Learned preferences and weights from a historical case review."""
    signal_weights: dict[str, float] = Field(default_factory=dict)
    interaction_effects: dict[str, float] = Field(default_factory=dict)
    risk_tolerance: float = 0.5  # 0 to 1 scale
    precision_vs_recall_preference: str = "balanced" # precision_heavy, recall_heavy, balanced
    reward_explanation: str = ""


class RewardScoreOutput(BaseModel):
    """Result of scoring the current decision against historical reward representations."""
    reward_score: float = Field(ge=0.0, le=1.0)
    confidence_alignment: float = Field(ge=0.0, le=1.0)
    decision_alignment: float = Field(ge=0.0, le=1.0)
    top_positive_signals: list[str] = Field(default_factory=list)
    top_negative_signals: list[str] = Field(default_factory=list)
    interaction_effects: list[str] = Field(default_factory=list)
    reward_explanation: str
