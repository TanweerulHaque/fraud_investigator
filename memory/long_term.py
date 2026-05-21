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
        reward_representation: dict[str, Any] | None = None,
    ) -> None:
        import json
        meta = {
            "decision": decision,
            "primary_risk_tags": str(case.primary_risk_tags),
            "reviewer_label": case.reviewer_label.value if case.reviewer_label else "",
            "reviewer_notes": explanation.get("analyst_rationale", "")[:300],
        }
        if reward_representation:
            meta["reward_representation"] = json.dumps(reward_representation)
            
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

    def get_all_cases_with_rewards(self) -> list[dict[str, Any]]:
        import json
        sqlite_cases = self.store.get_all_cases()
        chroma_res = self.vectors.get_all_metadata()
        chroma_map = {}
        if chroma_res and "ids" in chroma_res and "metadatas" in chroma_res:
            for cid, meta in zip(chroma_res["ids"], chroma_res["metadatas"]):
                chroma_map[cid] = meta

        for case in sqlite_cases:
            meta = chroma_map.get(case["case_id"])
            if meta and "reward_representation" in meta:
                try:
                    case["reward_representation"] = json.loads(meta["reward_representation"])
                except Exception:
                    case["reward_representation"] = None
            else:
                case["reward_representation"] = None
        return sqlite_cases
