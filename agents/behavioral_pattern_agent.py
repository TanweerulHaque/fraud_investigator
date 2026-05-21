"""Behavioral Pattern Agent — LLM interpretation of user behavior."""

from __future__ import annotations

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput, BehavioralStructured
from schemas.case import BehavioralSummary
from schemas.state import InvestigatorState
from utils.logging import get_logger

logger = get_logger(__name__)


class BehavioralPatternAgent(BaseAgent):
    name = "behavioral_pattern"
    system_prompt = (
        "You are a fraud investigator analyzing behavioral signals. "
        "Respond with ONLY a JSON object with keys: "
        "summary (string), risk_indicators (array of objects with signal, severity, detail, weight), "
        "stability_notes (string), "
        "stable_signals (array of plain strings — signal names only), "
        "drifting_signals (array of plain strings — signal names only). "
        "severity: low|medium|high. weight: 0-1. No markdown. No action recommendations."
    )

    def run(self, state: InvestigatorState) -> dict:
        logger.info("[BehavioralPattern] Starting run")
        case = state.get("normalized_case")
        if not case:
            return {"error": "no normalized case"}

        llm_raw = self.invoke_json(
            f"Analyze behavioral patterns:\n{self._case_payload(state)}\n\n"
            f"Retrieved cases:\n{state.get('retrieved_cases', [])}\n\n"
            f"Prior outputs:\n{self._prior_outputs(state)}"
        )

        indicators = self.parse_risk_indicators(llm_raw.get("risk_indicators"))

        # LLMs often put full indicator objects in drifting_signals — normalize to strings
        stable = self.coerce_str_list(llm_raw.get("stable_signals"))
        drifting = self.coerce_str_list(llm_raw.get("drifting_signals"))

        behavioral = BehavioralSummary(
            summary=str(llm_raw.get("summary") or ""),
            risk_indicators=indicators,
            stability_notes=str(llm_raw.get("stability_notes") or ""),
            stable_signals=stable,
            drifting_signals=drifting,
        )

        structured = BehavioralStructured(
            summary=behavioral.summary,
            risk_indicators=[i.model_dump() for i in indicators],
            stability_notes=behavioral.stability_notes,
            stable_signals=stable,
            drifting_signals=drifting,
        )
        dual = AgentDualOutput(agent_name=self.name, raw=llm_raw, structured=structured.model_dump())
        logger.info("[BehavioralPattern] Completed run (%d indicators)", len(indicators))
        return self.merge_state(state, {"behavioral_summary": behavioral}, dual)


def run_behavioral_pattern(state: InvestigatorState) -> dict:
    return BehavioralPatternAgent().run(state)
