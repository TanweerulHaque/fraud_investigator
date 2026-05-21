"""Explanation Agent — analyst rationale via LLM."""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput, ExplanationStructured
from schemas.case import ExplanationOutput
from schemas.state import InvestigatorState


class ExplanationAgent(BaseAgent):
    name = "explanation"
    system_prompt = (
        "You write an internal fraud analyst review note grounded in evidence. "
        "Return JSON: analyst_rationale (string), top_reasons (list), risk_narrative (string). "
        "Only use facts from the provided context. No customer action recommendations."
    )

    def run(self, state: InvestigatorState) -> dict:
        if not state.get("decision"):
            return {"error": "missing decision for explanation"}

        llm_raw = self.invoke_json(
            f"Write analyst explanation:\n"
            f"Decision: {state.get('decision')} ({state.get('confidence')})\n\n"
            f"Case:\n{self._case_payload(state)}\n\n"
            f"Critic:\n{state.get('critic_output')}\n\n"
            f"All agent outputs:\n{self._prior_outputs(state)}"
        )

        explanation = ExplanationOutput(
            analyst_rationale=llm_raw.get("analyst_rationale", ""),
            top_reasons=llm_raw.get("top_reasons") or state.get("decision_drivers") or [],
            risk_narrative=llm_raw.get("risk_narrative", ""),
        )
        structured = ExplanationStructured(
            analyst_rationale=explanation.analyst_rationale,
            top_reasons=explanation.top_reasons,
            risk_narrative=explanation.risk_narrative,
        )
        dual = AgentDualOutput(agent_name=self.name, raw=llm_raw, structured=structured.model_dump())
        return self.merge_state(state, {"explanation_output": explanation}, dual)


def run_explanation(state: InvestigatorState) -> dict:
    return ExplanationAgent().run(state)
