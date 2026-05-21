"""Agent to compute behavioral alignment score based on historical reward models."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput
from schemas.case import RewardScoreOutput
from schemas.state import InvestigatorState
from utils.logging import get_logger

logger = get_logger(__name__)


class RewardScoringAgent(BaseAgent):
    name = "reward_scoring"
    system_prompt = (
        "You are an expert fraud investigator. Your task is to calculate a 'Reward Score' (0.0 to 1.0) "
        "that represents how closely the proposed decision draft aligns with the historical reviewer's cognitive preferences.\n"
        "You will be given the current case details, the draft decision, and the historical reward representations.\n"
        "1. Identify which historical variables and interaction effects are present in the current case.\n"
        "2. Compute if the draft decision aligns with the historical risk tolerance.\n"
        "3. Output a single JSON matching the RewardScoreOutput schema:\n"
        "- reward_score: float (0.0 to 1.0)\n"
        "- confidence_alignment: float (0.0 to 1.0)\n"
        "- decision_alignment: float (0.0 to 1.0)\n"
        "- top_positive_signals: list of strings (signals that matched historical preferences)\n"
        "- top_negative_signals: list of strings (signals that historically indicate the opposite)\n"
        "- interaction_effects: list of strings describing any nonlinear effects found.\n"
        "- reward_explanation: string explaining the reward score."
    )

    def run(self, state: InvestigatorState) -> dict[str, Any]:
        draft = state.get("decision_draft")
        reward_outputs = state.get("agent_outputs", {}).get("reward_retrieval", {})
        
        if not draft:
            logger.warning("No decision draft found for reward scoring.")
            return {}

        logger.info(f"Reward scoring inputs - draft: {draft}, reward_outputs: {reward_outputs}")

        prompt = (
            f"Case Payload:\n{self._case_payload(state)}\n\n"
            f"Draft Decision: {draft.model_dump_json() if hasattr(draft, 'model_dump_json') else draft}\n\n"
            f"Historical Reward Synthesis:\n{reward_outputs}\n\n"
            "Score the draft decision against the historical preferences. Return a JSON matching RewardScoreOutput."
        )

        try:
            logger.info(f"Reward scoring prompt: {prompt}")
            raw_json = self.invoke_json(prompt)
            logger.info(f"Reward scoring raw output: {raw_json}")

            dual = self.build_dual_output(raw_json, RewardScoreOutput)
            logger.info(f"Reward scoring dual output: {dual}")
            data = self.merge_state(state, {"reward_score_output": dual.structured}, dual)
            logger.info(f"Reward scoring merged state: {data}")

            return data
        except Exception as e:
            logger.error("Reward scoring failed: %s", e)
            return {"error": f"Reward scoring failed: {e}"}

def run_reward_scoring(state: InvestigatorState) -> dict[str, Any]:
    return RewardScoringAgent().run(state)
