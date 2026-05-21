"""ChromaDB similarity search over indexed fraud cases."""

from __future__ import annotations

from typing import Any

from memory.vector_store import VectorStore
from schemas.case import NormalizedCase, RetrievedCase
from utils.config import get_config


class SimilaritySearchTool:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.store = vector_store or VectorStore()

    def search(
        self,
        query_vector: list[float],
        exclude_case_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedCase]:
        cfg = get_config()
        k = top_k or cfg["retrieval"]["top_k"]
        min_sim = cfg["retrieval"]["min_similarity"]
        hits = self.store.search(query_vector, top_k=k, exclude_case_id=exclude_case_id)
        results: list[RetrievedCase] = []
        for case_id, sim, meta in hits:
            if sim < min_sim:
                continue
            tags = meta.get("primary_risk_tags", "")
            if isinstance(tags, str) and tags.startswith("["):
                import json

                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError:
                    tags = []
            reward_rep = None
            if "reward_representation" in meta:
                import json
                try:
                    reward_rep = json.loads(meta["reward_representation"])
                except json.JSONDecodeError:
                    reward_rep = None

            results.append(
                RetrievedCase(
                    case_id=case_id,
                    similarity=round(sim, 4),
                    decision=meta.get("decision"),
                    primary_risk_tags=tags if isinstance(tags, list) else [],
                    reviewer_notes_snippet=str(meta.get("reviewer_notes", ""))[:200],
                    reward_representation=reward_rep,
                )
            )
        return results

    def index_case(
        self,
        case: NormalizedCase,
        case_summary: str,
        metadata: dict[str, Any],
    ) -> list[float]:
        meta = {**metadata, "case_id": case.case_id}
        return self.store.upsert(case.case_id, case, case_summary, meta)
