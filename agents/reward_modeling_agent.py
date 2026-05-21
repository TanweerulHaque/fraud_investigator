"""Agent to learn preference weights and interaction effects from reviewer behavior."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from schemas.agent_output import AgentDualOutput
from schemas.case import RewardRepresentation
from schemas.state import InvestigatorState
from utils.logging import get_logger

logger = get_logger(__name__)


class RewardModelingAgent(BaseAgent):
    name = "reward_modeling"
    system_prompt = (
        "You are an expert fraud investigator studying a colleague's historical review. "
        "Your task is to reverse-engineer their cognitive preferences into a set of structured weights and penalties. "
        "Analyze the case details along with the final decision and their explanation notes. "
        "Output a valid JSON matching the RewardRepresentation schema with:\n"
        "- signal_weights: dict of variable name to float weight (positive for trust/safe, negative for fraud/risk).\n"
        "- interaction_effects: dict of interaction strings to float weight (e.g. 'gambling_apps & night_activity': -1.5).\n"
        "- risk_tolerance: float (0.0=extremely risk averse, 1.0=very tolerant).\n"
        "- precision_vs_recall_preference: string ('precision_heavy', 'recall_heavy', 'balanced').\n"
        "- reward_explanation: string explaining why you inferred these weights.\n"
        "You MUST output exactly ONE valid JSON object."
    )

    def run(self, state: InvestigatorState, historical_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        case = state.get("normalized_case")
        raw = state.get("raw_case", {})
        
        if not case:
            return {"error": "Missing normalized_case in state"}

        # We can extract reward from historical labels if available
        reviewer_label = case.reviewer_label.value if case.reviewer_label else raw.get("reviewer_label")
        reviewer_notes = raw.get("metadata", {}).get("reviewer_notes", "")
        
        if not reviewer_label and not reviewer_notes:
            logger.info("No human reviewer data present, skipping reward modeling.")
            return {}

        historical_str = ""
        if historical_cases:
            historical_str = "Here are the previous historical cases and their human decisions, notes, and derived reward representations for context:\n"
            for hc in historical_cases:
                hc_id = hc.get("case_id", "unknown")
                hc_decision = hc.get("reviewer_label") or hc.get("decision", "")
                hc_notes = hc.get("explanation", {}).get("analyst_rationale", "")
                hc_rep = hc.get("reward_representation")
                historical_str += f"- Case {hc_id}:\n"
                historical_str += f"  Decision: {hc_decision}\n"
                historical_str += f"  Notes: {hc_notes}\n"
                if hc_rep:
                    historical_str += f"  Previous Derived Weights/Penalties: {hc_rep}\n"
                historical_str += "\n"

        prompt = ""
        if historical_str:
            prompt += f"{historical_str}\n"

        prompt += (
            f"Current Case Payload to analyze:\n{self._case_payload(state)}\n\n"
            f"Current Human Reviewer Decision: {reviewer_label}\n"
            f"Current Human Reviewer Notes: {reviewer_notes}\n\n"
            "Extract the reward weights and preferences for the current case as a JSON object, taking the previous historical cases and their decisions/weights into account to ensure consistent and unified cognitive preference modeling."
        )

        try:
            raw_json = self.invoke_json(prompt)
            logger.info(f"Reward modeling raw output: {raw_json}")

            dual = self.build_dual_output(raw_json, RewardRepresentation)
            logger.info(f"Reward modeling dual output: {dual}")

            # Extract structured model and save to state
            reward_rep = dual.structured

            # This agent typically runs inside memory_update or seeding, so we don't strictly merge it 
            # into the standard state unless needed. But we can return the payload so caller can use it.
            return {
                "reward_representation": reward_rep,
                "agent_outputs": {self.name: dual.model_dump()}
            }
        except Exception as e:
            logger.error("Reward modeling failed: %s", e)
            return {"error": f"Reward modeling failed: {e}"}

def run_reward_modeling(state: InvestigatorState, historical_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return RewardModelingAgent().run(state, historical_cases)
