"""Critic Agent — counterfactuals and uncertainty via LLM."""

from __future__ import annotations

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput, CriticStructured
from schemas.case import CriticOutput
from schemas.state import InvestigatorState
from utils.logging import get_logger

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    name = "critic"
    system_prompt = (
        "You challenge a fraud investigator's provisional decision. "
        "Respond with ONLY a single JSON object (no markdown fences), with keys: "
        "contradictory_evidence (array of strings), counterfactuals (array of strings), "
        "uncertainty_notes (array of strings), reasoning (string). "
        "Each array element must be a plain string sentence, not an object. "
        "Reference historical pass/fail patterns. No action recommendations."
    )

    def run(self, state: InvestigatorState) -> dict:
        logger.info("[Critic] Starting run")
        if not state.get("decision"):
            return {"error": "missing decision for critic"}

        llm_raw = self.invoke_json(
            f"Challenge this judgment:\n"
            f"Decision: {state.get('decision')}\n"
            f"Confidence: {state.get('confidence')}\n"
            f"Drivers: {state.get('decision_drivers')}\n\n"
            f"Case:\n{self._case_payload(state)}\n\n"
            f"Behavioral:\n{state.get('behavioral_summary')}\n\n"
            f"Prior outputs:\n{self._prior_outputs(state)}"
        )
        logger.info("[Critic] parsed keys: %s", list(llm_raw.keys()))

        critic = CriticOutput(
            contradictory_evidence=self.coerce_str_list(llm_raw.get("contradictory_evidence")),
            counterfactuals=self.coerce_str_list(llm_raw.get("counterfactuals")),
            uncertainty_notes=self.coerce_str_list(llm_raw.get("uncertainty_notes")),
        )
        structured = CriticStructured(
            contradictory_evidence=critic.contradictory_evidence,
            counterfactuals=critic.counterfactuals,
            uncertainty_notes=critic.uncertainty_notes,
            reasoning=str(llm_raw.get("reasoning") or ""),
        )
        dual = AgentDualOutput(agent_name=self.name, raw=llm_raw, structured=structured.model_dump())
        logger.info("[Critic] Completed run")
        return self.merge_state(state, {"critic_output": critic}, dual)


def run_critic(state: InvestigatorState) -> dict:
    return CriticAgent().run(state)
