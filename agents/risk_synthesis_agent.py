"""Risk Synthesis Agent — fraud / not_fraud judgment via LLM."""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput, RiskSynthesisStructured
from schemas.case import DecisionDraft, ReviewerDecision
from schemas.state import InvestigatorState


class RiskSynthesisAgent(BaseAgent):
    name = "risk_synthesis"
    system_prompt = (
        "You are a senior fraud investigator rendering a binary judgment. "
        "Decision MUST be exactly 'fraud' or 'not_fraud' (never needs_review). "
        "Return JSON: decision (fraud|not_fraud), confidence (0-1), "
        "decision_drivers (list of strings), reasoning (string). "
        "Use evidence, similar cases, and behavioral analysis. No action recommendations."
    )

    def run(self, state: InvestigatorState) -> dict:
        if not state.get("normalized_case") or not state.get("behavioral_summary"):
            return {"error": "missing inputs for risk synthesis"}

        logger.info("[RiskSynthesis] Starting run")
        llm_raw = self.invoke_json(
            f"Synthesize fraud judgment:\n{self._case_payload(state)}\n\n"
            f"Behavioral:\n{state.get('behavioral_summary')}\n\n"
            f"Retrieved:\n{state.get('retrieved_cases', [])}\n\n"
            f"Quality flags: {state.get('quality_flags')}\n\n"
            f"Prior outputs:\n{self._prior_outputs(state)}"
        )

        decision_str = str(llm_raw.get("decision", "not_fraud")).lower().strip()
        if decision_str not in ("fraud", "not_fraud"):
            decision_str = "fraud" if "fraud" in decision_str else "not_fraud"

        decision = ReviewerDecision(decision_str)
        confidence = float(llm_raw.get("confidence", 0.5))
        drivers = llm_raw.get("decision_drivers") or []

        structured = RiskSynthesisStructured(
            decision=decision_str,
            confidence=confidence,
            decision_drivers=drivers,
            reasoning=llm_raw.get("reasoning", ""),
        )
        draft = DecisionDraft(
            decision=decision,
            confidence=round(confidence, 3),
            decision_drivers=drivers[:12],
        )
        dual = AgentDualOutput(agent_name=self.name, raw=llm_raw, structured=structured.model_dump())
        logger.info("[RiskSynthesis] Completed run")
        data = self.merge_state(
            state,
            {
                "decision_draft": draft,
                "decision": decision,
                "confidence": draft.confidence,
                "decision_drivers": draft.decision_drivers,
            },
            dual,
        )
        logger.info(f"Merged state: {data}")
        return data


def run_risk_synthesis(state: InvestigatorState) -> dict:
    return RiskSynthesisAgent().run(state)
