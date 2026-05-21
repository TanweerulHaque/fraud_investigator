"""Agent to retrieve and synthesize historical reward representations."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from schemas.state import InvestigatorState
from utils.logging import get_logger

logger = get_logger(__name__)


class RewardRetrievalAgent(BaseAgent):
    name = "reward_retrieval"
    system_prompt = (
        "You are an expert fraud investigator. You will be provided with several historical "
        "reward representations (weights, interaction effects, risk tolerances) extracted from similar past cases.\n"
        "Your job is to synthesize these into a single unified list of important weights and interaction effects "
        "that should be applied to evaluate the current case.\n"
        "Output a valid JSON containing:\n"
        "- synthesized_weights: dict of variable name to float weight\n"
        "- synthesized_interactions: dict of interaction to float weight\n"
        "- risk_tolerance: float\n"
        "- retrieval_analysis: string explaining what the dominant historical preferences are."
    )

    def run(self, state: InvestigatorState) -> dict[str, Any]:
        retrieved_cases = state.get("retrieved_cases", [])
        
        valid_reps = [
            c.reward_representation for c in retrieved_cases 
            if getattr(c, "reward_representation", None)
        ]
        
        if not valid_reps:
            logger.info("No historical reward representations found in retrieved cases.")
            return {"reward_representations": []}

        logger.info(f"Found {valid_reps} historical reward representations")

        import json
        prompt = (
            f"Here are the historical reward representations from {len(valid_reps)} similar cases:\n"
            f"{json.dumps(valid_reps, indent=2)}\n\n"
            "Synthesize them into a single JSON object."
        )

        try:
            raw_json = self.invoke_json(prompt)
            logger.info(f"Reward retrieval raw output: {raw_json}")

            # Store both the raw representations and the synthesized one
            return {
                "reward_representations": valid_reps,
                "agent_outputs": {self.name: {"raw": raw_json, "structured": raw_json}}
            }
        except Exception as e:
            logger.error("Reward retrieval synthesis failed: %s", e)
            return {"reward_representations": valid_reps}

def run_reward_retrieval(state: InvestigatorState) -> dict[str, Any]:
    return RewardRetrievalAgent().run(state)
