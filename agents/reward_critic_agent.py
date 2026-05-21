"""Agent to check whether the decision violates historical reviewer preferences."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput
from schemas.state import InvestigatorState
from utils.logging import get_logger

logger = get_logger(__name__)


class RewardCriticOutput(BaseModel):
    preference_violations: list[str] = Field(default_factory=list)
    suggested_adjustments: list[str] = Field(default_factory=list)
    critic_notes: str


class RewardCriticAgent(BaseAgent):
    name = "reward_critic"
    system_prompt = (
        "You are an expert fraud investigator verifying a draft decision against historical preferences.\n"
        "You will be given the draft decision, the Reward Score output, and the historical reward representations.\n"
        "Check if the decision violates the established historical preferences (e.g., historical reviewer strongly weights "
        "stable_device but the draft ignored it).\n"
        "Output a valid JSON matching the RewardCriticOutput schema:\n"
        "- preference_violations: list of strings (where the draft diverges from historical behavior)\n"
        "- suggested_adjustments: list of strings (how to better align with the historical reviewer)\n"
        "- critic_notes: string (summary of your critique)"
    )

    def run(self, state: InvestigatorState) -> dict[str, Any]:
        draft = state.get("decision_draft")
        reward_score = state.get("reward_score_output")
        reward_retrieval = state.get("agent_outputs", {}).get("reward_retrieval", {})

        if not draft or not reward_score:
            logger.warning("Missing draft or reward score for reward critic.")
            return {}

        prompt = (
            f"Draft Decision:\n{draft.model_dump_json() if hasattr(draft, 'model_dump_json') else draft}\n\n"
            f"Reward Score:\n{reward_score}\n\n"
            f"Historical Preferences:\n{reward_retrieval}\n\n"
            "Generate a critique based on historical preference alignment."
        )

        try:
            raw_json = self.invoke_json(prompt)
            logger.info(f"Reward critic raw output: {raw_json}")

            dual = self.build_dual_output(raw_json, RewardCriticOutput)
            logger.info(f"Reward critic dual output: {dual}")

            data = self.merge_state(state, {"reward_critic_output": dual.structured}, dual)
            logger.info(f"Reward critic data: {data}")

            return data
        except Exception as e:
            logger.error("Reward critic failed: %s", e)
            return {"error": f"Reward critic failed: {e}"}

def run_reward_critic(state: InvestigatorState) -> dict[str, Any]:
    return RewardCriticAgent().run(state)
