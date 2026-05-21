"""Long-term case memory facade (SQLite + Chroma)."""

from __future__ import annotations

from typing import Any

from memory.vector_store import VectorStore
from schemas.case import NormalizedCase
from tools.similarity import SimilaritySearchTool
from tools.sqlite_store import CaseStore


class LongTermMemory:
    def __init__(
        self,
        store: CaseStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.store = store or CaseStore()
        self.vectors = vector_store or VectorStore()
        self.search_tool = SimilaritySearchTool(self.vectors)

    def remember_investigation(
        self,
        case: NormalizedCase,
        payload: dict[str, Any],
        case_summary: str,
        query_vector: list[float],
        decision: str,
        confidence: float,
        explanation: dict[str, Any],
    ) -> None:
        meta = {
            "decision": decision,
            "primary_risk_tags": str(case.primary_risk_tags),
            "reviewer_label": case.reviewer_label.value if case.reviewer_label else "",
            "reviewer_notes": explanation.get("analyst_rationale", "")[:300],
        }
        self.search_tool.index_case(case, case_summary, meta)
        self.store.upsert_case(
            case_id=case.case_id,
            payload=payload,
            decision=decision,
            confidence=confidence,
            primary_risk_tags=case.primary_risk_tags,
            reviewer_label=meta.get("reviewer_label") or None,
            explanation=explanation,
        )

    def retrieve_similar(
        self,
        query_vector: list[float],
        exclude_case_id: str | None = None,
        top_k: int = 2,
    ):
        return self.search_tool.search(query_vector, exclude_case_id=exclude_case_id, top_k=top_k)
