"""Embedding Agent — case summary (LLM) + vector for Chroma retrieval."""

from __future__ import annotations

from agents.base import BaseAgent
from memory.vector_store import VectorStore
from schemas.agent_output import AgentDualOutput, EmbeddingStructured
from schemas.state import InvestigatorState

import logging
logger = logging.getLogger(__name__)


class EmbeddingAgent(BaseAgent):
    name = "embedding"
    system_prompt = (
        "You are a fraud investigator summarizing a case for semantic retrieval. "
        "Write a dense factual summary (no recommendations). "
        "Return JSON: case_summary (string, 3-6 sentences)."
    )

    def run(self, state: InvestigatorState) -> dict:
        case = state.get("normalized_case")
        if not case:
            return {"error": "no normalized case for embedding"}

        logger.info("Embedding agent running to summarise the case...")
        message_to_summarise = f"""
Understand the decision of this case by a reviewer. summarize this case in english. keep it short and concise:

\"\"\"
{self._case_payload(state)}
Prior agent outputs:
{self._prior_outputs(state)}
\"\"\"
"""
        logger.info(f"Message to summarise: {message_to_summarise}")

        llm_raw = self.invoke_json(message_to_summarise)
        logger.info(f"Agent summarisation: {llm_raw}")

        case_summary = llm_raw.get("case_summary", "")
        logger.info(f"Case summary: {case_summary}")
        if not case_summary:
            return {"error": "embedding agent failed to produce case_summary"}

        logger.info("Building combined vector...")
        store = VectorStore()
        vector = store.build_combined_vector(case, case_summary)
        feat_dim = len(store.handcrafted_features(case))
        summary_dim = len(vector) - feat_dim

        structured = EmbeddingStructured(
            case_id=case.case_id,
            case_summary=case_summary,
            feature_dim=feat_dim,
            summary_embedding_dim=summary_dim,
            combined_dim=len(vector),
        )
        dual = AgentDualOutput(
            agent_name=self.name,
            raw={**llm_raw, "embedding_preview_dims": len(vector)},
            structured=structured.model_dump(),
        )
        data = self.merge_state(
            state,
            {"case_summary": case_summary, "case_embedding": vector.tolist()},
            dual,
        )
        logger.info(f"Merged state: {data}")
        return data


def run_embedding(state: InvestigatorState) -> dict:
    return EmbeddingAgent().run(state)
