"""Evidence Retrieval Agent — Chroma similarity + LLM interpretation."""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from memory.long_term import LongTermMemory
from schemas.agent_output import AgentDualOutput, RetrievalStructured
from schemas.state import InvestigatorState


class EvidenceRetrievalAgent(BaseAgent):
    name = "evidence_retrieval"
    system_prompt = (
        "You interpret similar historical fraud cases retrieved from vector search. "
        "Return JSON: matched_patterns (list of strings), retrieval_analysis (string), "
        "lessons_from_history (string describing why similar cases passed or failed)."
    )

    def __init__(self, long_term: LongTermMemory) -> None:
        super().__init__()
        self.long_term = long_term

    def run(self, state: InvestigatorState) -> dict:
        logger.info("[EvidenceRetrieval] Starting run")
        case = state.get("normalized_case")
        embedding = state.get("case_embedding")
        if not case or not embedding:
            return {"error": "missing case or embedding for retrieval"}

        retrieved = self.long_term.retrieve_similar(
            query_vector=embedding,
            exclude_case_id=case.case_id,
        )

        retrieved_json = [r.model_dump() for r in retrieved]
        llm_raw = self.invoke_json(
            f"Current case:\n{case.model_dump_json()}\n\n"
            f"Retrieved similar cases from Chroma:\n{retrieved_json}\n\n"
            f"Explain patterns and what past pass/fail outcomes imply."
        )

        patterns = llm_raw.get("matched_patterns") or []
        snippets = [
            f"{r.case_id}: {r.reviewer_notes_snippet}" for r in retrieved if r.reviewer_notes_snippet
        ]
        fraud_votes = sum(1 for r in retrieved if r.decision == "fraud")
        not_fraud_votes = sum(1 for r in retrieved if r.decision == "not_fraud")

        structured = RetrievalStructured(
            retrieved_count=len(retrieved),
            matched_patterns=patterns,
            neighbor_fraud_votes=fraud_votes,
            neighbor_not_fraud_votes=not_fraud_votes,
        )
        dual = AgentDualOutput(
            agent_name=self.name,
            raw={**llm_raw, "retrieved_cases": retrieved_json},
            structured=structured.model_dump(),
        )
        logger.info(f"[EvidenceRetrieval] Retrieved {len(retrieved)} cases, patterns: {len(patterns)}")
        result = self.merge_state(
            state,
            {
                "retrieved_cases": retrieved,
                "matched_patterns": patterns,
                "prior_reasoning_snippets": snippets[:5],
            },
            dual,
        )
        logger.info("[EvidenceRetrieval] Completed run")
        return result


def run_evidence_retrieval(state: InvestigatorState, long_term: LongTermMemory) -> dict:
    return EvidenceRetrievalAgent(long_term).run(state)
