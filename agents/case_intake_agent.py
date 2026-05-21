"""Case Intake Agent — validate structure; LLM quality assessment."""

from __future__ import annotations

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput, IntakeStructured
from schemas.case import NormalizedCase
from schemas.state import InvestigatorState
from tools.validator import validate_case_dict
import logging
logger = logging.getLogger(__name__)


class CaseIntakeAgent(BaseAgent):
    name = "case_intake"
    system_prompt = (
        "You are a fraud case intake specialist. Assess data quality for investigation. "
        "Respond with ONLY a JSON object with keys: "
        "quality_flags (array of strings), missing_fields (array of strings), "
        "intake_notes (string). No markdown. Do not decide fraud/not_fraud."
    )

    def run(self, state: InvestigatorState) -> dict:
        logger.info("[CaseIntake] Starting run")
        raw = state.get("raw_case") or {}
        missing: list[str] = []
        if not raw.get("transactions"):
            missing.append("transactions")
        if not raw.get("installed_apps"):
            missing.append("installed_apps")
        logger.info(f"[CaseIntake] Missing fields: {missing}")

        fraud_case, schema_err, pydantic_err = validate_case_dict(raw)
        if fraud_case is None:
            logger.error(f"[CaseIntake] Validation failed: {schema_err + pydantic_err}")
            return {
                "error": f"validation failed: {schema_err + pydantic_err}",
                "missing_fields": missing,
                "quality_flags": ["invalid_case"],
            }

        normalized = NormalizedCase.from_fraud_case(fraud_case)

        llm_raw = self.invoke_json(
            f"Assess intake quality for this case JSON:\n{normalized.model_dump_json()}\n"
            f"Already detected missing: {missing}"
        )
        logger.info(f"[CaseIntake] LLM raw response: {llm_raw}")
        flags = list(
            set(self.coerce_str_list(llm_raw.get("quality_flags"))
                + (["sparse_transaction_history"] if "transactions" in missing else []))
        )
        missing_fields = list(
            set(self.coerce_str_list(llm_raw.get("missing_fields")) + missing)
        )

        structured = IntakeStructured(
            case_id=normalized.case_id,
            quality_flags=flags,
            missing_fields=missing_fields,
            intake_notes=llm_raw.get("intake_notes", ""),
        )
        dual = AgentDualOutput(
            agent_name=self.name,
            raw=llm_raw,
            structured=structured.model_dump(),
        )
        result = self.merge_state(
            state,
            {
                "normalized_case": normalized,
                "missing_fields": missing_fields,
                "quality_flags": flags,
            },
            dual,
        )
        logger.info("[CaseIntake] Completed run")
        return result


def run_case_intake(state: InvestigatorState) -> dict:
    return CaseIntakeAgent().run(state)
