"""Case Intake Agent — validate structure; LLM quality assessment."""

from __future__ import annotations

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput, IntakeStructured
from schemas.case import NormalizedCase
from schemas.state import InvestigatorState
from tools.validator import validate_case_dict
from utils.logging import get_logger

logger = get_logger(__name__)

# Computed at normalization or used only for eval — not required on raw intake payload
_INTAKE_OPTIONAL_FIELDS = frozenset({
    "reviewer_label",
    "primary_risk_tags",
    "high_risk_app_count",
    "prior_dispute_details",
    "total_txn_amount_7d",
    "cash_out_ratio",
})


class CaseIntakeAgent(BaseAgent):
    name = "case_intake"
    system_prompt = (
        "You are a fraud case intake specialist. Assess whether the RAW case payload "
        "has enough data to investigate (transactions, device, model scores, profile). "
        "Respond with ONLY a JSON object: "
        "quality_flags (array of strings), missing_fields (array of strings — only "
        "missing INPUT fields from the raw JSON, not derived/computed fields), "
        "intake_notes (string, never null; use empty string if none). "
        "Do not flag reviewer_label, primary_risk_tags, or high_risk_app_count as missing. "
        "Do not decide fraud/not_fraud."
    )

    def run(self, state: InvestigatorState) -> dict:
        logger.info("[CaseIntake] Starting run")
        raw = state.get("raw_case") or {}

        logger.info(f"Raw case data: {raw}")

        missing: list[str] = []
        if not raw.get("transactions"):
            missing.append("transactions")
        if not raw.get("installed_apps"):
            missing.append("installed_apps")

        fraud_case, schema_err, pydantic_err = validate_case_dict(raw)
        if fraud_case is None:
            logger.error("[CaseIntake] Validation failed: %s", schema_err + pydantic_err)
            return {
                "error": f"validation failed: {schema_err + pydantic_err}",
                "missing_fields": missing,
                "quality_flags": ["invalid_case"],
            }

        normalized = NormalizedCase.from_fraud_case(fraud_case)

        llm_raw = self.invoke_json(
            f"Assess intake quality for this RAW case JSON (investigation input only):\n"
            f"{raw}\n\n"
            f"Normalized derived fields (do NOT report as missing): "
            f"primary_risk_tags={normalized.primary_risk_tags}, "
            f"high_risk_app_count={normalized.high_risk_app_count}\n"
            f"Already detected missing required sections: {missing}"
        )
        logger.info("[CaseIntake] LLM raw response keys: %s", list(llm_raw.keys()))

        flags = list(
            set(self.coerce_str_list(llm_raw.get("quality_flags"))
                + (["sparse_transaction_history"] if "transactions" in missing else []))
        )
        llm_missing = [
            m for m in self.coerce_str_list(llm_raw.get("missing_fields"))
            if m not in _INTAKE_OPTIONAL_FIELDS
        ]
        missing_fields = list(set(llm_missing + missing))

        intake_notes = str(llm_raw.get("intake_notes") or "").strip()

        structured = IntakeStructured(
            case_id=normalized.case_id,
            quality_flags=flags,
            missing_fields=missing_fields,
            intake_notes=intake_notes,
        )
        dual = AgentDualOutput(
            agent_name=self.name,
            raw=llm_raw,
            structured=structured.model_dump(),
        )
        return self.merge_state(
            state,
            {
                "normalized_case": normalized,
                "missing_fields": missing_fields,
                "quality_flags": flags,
            },
            dual,
        )


def run_case_intake(state: InvestigatorState) -> dict:
    return CaseIntakeAgent().run(state)
