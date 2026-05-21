"""Memory Update Agent — persist to SQLite + Chroma with embedding agent output."""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from memory.long_term import LongTermMemory
from schemas.agent_output import AgentDualOutput, MemoryUpdateStructured
from schemas.state import InvestigatorState


class MemoryUpdateAgent(BaseAgent):
    name = "memory_update"
    system_prompt = (
        "You confirm what was stored in investigator memory for audit. "
        "Return JSON: storage_rationale (string), lessons_for_future_cases (string)."
    )

    def __init__(self, long_term: LongTermMemory) -> None:
        super().__init__()
        self.long_term = long_term

    def run(self, state: InvestigatorState) -> dict:
        case = state.get("normalized_case")
        decision = state.get("decision")
        embedding = state.get("case_embedding")
        summary = state.get("case_summary")
        explanation = state.get("explanation_output")
        raw = state.get("raw_case") or {}

        if not case or not decision or not embedding or not summary:
            dual = AgentDualOutput(
                agent_name=self.name,
                raw={"status": "skipped"},
                structured=MemoryUpdateStructured(case_id=case.case_id if case else "", stored=False).model_dump(),
            )
            return self.merge_state(state, {"memory_updated": False}, dual)

        expl_dict = explanation.model_dump() if explanation else {}
        self.long_term.remember_investigation(
            case=case,
            payload=raw,
            case_summary=summary,
            query_vector=embedding,
            decision=decision.value,
            confidence=state.get("confidence") or 0.0,
            explanation=expl_dict,
        )

        llm_raw = self.invoke_json(
            f"Case {case.case_id} stored with decision {decision.value}. "
            f"Summary length {len(summary)}. Confirm audit log rationale."
        )

        label = case.reviewer_label.value if case.reviewer_label else None
        if label:
            was_correct = label == decision.value
            self.long_term.store.log_review_outcome(
                case.case_id, decision.value, label, was_correct, state.get("confidence") or 0.0
            )

        structured = MemoryUpdateStructured(
            case_id=case.case_id,
            stored=True,
            chroma_id=case.case_id,
        )
        dual = AgentDualOutput(
            agent_name=self.name,
            raw=llm_raw,
            structured=structured.model_dump(),
        )
        return self.merge_state(state, {"memory_updated": True}, dual)


def run_memory_update(state: InvestigatorState, long_term: LongTermMemory) -> dict:
    return MemoryUpdateAgent(long_term).run(state)
