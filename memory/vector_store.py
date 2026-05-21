"""Chroma-backed case vectors: handcrafted features + summary embedding."""

from __future__ import annotations

from typing import Any

import chromadb
import numpy as np

from agents.embedding_factory import embed_text
from schemas.case import NormalizedCase
from utils.config import get_config, resolve_path

FEATURE_DIM = 16


class VectorStore:
    def __init__(self) -> None:
        cfg = get_config()
        chroma_path = str(resolve_path(cfg["paths"]["chroma_path"]))
        collection_name = cfg.get("chroma", {}).get("collection_name", "fraud_case_vectors")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def handcrafted_features(case: NormalizedCase) -> np.ndarray:
        txn_count = len(case.transactions)
        avg_amt = sum(t.amount for t in case.transactions) / txn_count if txn_count else 0.0
        features = [
            case.user_profile.account_age_days / 365.0,
            case.user_profile.prior_disputes / 5.0,
            case.cash_out_ratio,
            case.high_risk_app_count / 5.0,
            case.total_txn_amount_7d / 100000.0,
            case.app_usage.night_usage_ratio,
            case.app_usage.unique_locations_7d / 10.0,
            case.location_patterns.max_geo_velocity_kmph / 1000.0,
            len(case.location_patterns.countries_7d) / 5.0,
            float(case.device.is_emulator),
            float(case.device.is_rooted),
            case.device.device_changes_30d / 5.0,
            case.model_scores.fraud_score,
            case.model_scores.aml_score,
            case.model_scores.account_takeover_score,
            avg_amt / 50000.0,
            txn_count / 20.0,
        ]
        vec = np.array(features, dtype=np.float64)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def build_combined_vector(self, case: NormalizedCase, case_summary: str) -> np.ndarray:
        """Handcrafted features + embedding(summary) concatenated for Chroma."""
        feat = self.handcrafted_features(case)
        summary_emb = np.array(embed_text(case_summary), dtype=np.float64)
        se_norm = np.linalg.norm(summary_emb)
        if se_norm > 0:
            summary_emb = summary_emb / se_norm
        return np.concatenate([feat, summary_emb])

    def upsert(
        self,
        case_id: str,
        case: NormalizedCase,
        case_summary: str,
        metadata: dict[str, Any],
    ) -> list[float]:
        vector = self.build_combined_vector(case, case_summary)
        meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in metadata.items()}
        self.collection.upsert(
            ids=[case_id],
            embeddings=[vector.tolist()],
            documents=[case_summary],
            metadatas=[meta],
        )
        return vector.tolist()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        exclude_case_id: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if self.collection.count() == 0:
            return []

        n = min(top_k + (1 if exclude_case_id else 0), self.collection.count())
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            include=["metadatas", "distances", "documents"],
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        hits: list[tuple[str, float, dict[str, Any]]] = []
        for cid, dist, meta in zip(ids, distances, metadatas):
            similarity = 1.0 - float(dist)
            m = dict(meta or {})
            m.setdefault("case_id", cid)
            if exclude_case_id and cid == exclude_case_id:
                continue
            hits.append((cid, similarity, m))
        return hits
