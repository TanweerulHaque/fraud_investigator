"""SQLite persistence for cases and decisions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import get_config, resolve_path


class CaseStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_config()
        self.db_path = Path(db_path) if db_path else resolve_path(cfg["paths"]["db_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    decision TEXT,
                    confidence REAL,
                    primary_risk_tags TEXT,
                    reviewer_label TEXT,
                    explanation_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    twin_decision TEXT,
                    reviewer_label TEXT,
                    was_correct INTEGER,
                    confidence REAL,
                    logged_at TEXT NOT NULL
                );
                """
            )

    def upsert_case(
        self,
        case_id: str,
        payload: dict[str, Any],
        decision: str | None = None,
        confidence: float | None = None,
        primary_risk_tags: list[str] | None = None,
        reviewer_label: str | None = None,
        explanation: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        tags = json.dumps(primary_risk_tags or [])
        expl = json.dumps(explanation or {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cases (case_id, payload_json, decision, confidence,
                    primary_risk_tags, reviewer_label, explanation_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    decision=excluded.decision,
                    confidence=excluded.confidence,
                    primary_risk_tags=excluded.primary_risk_tags,
                    reviewer_label=excluded.reviewer_label,
                    explanation_json=excluded.explanation_json,
                    updated_at=excluded.updated_at
                """,
                (
                    case_id,
                    json.dumps(payload),
                    decision,
                    confidence,
                    tags,
                    reviewer_label,
                    expl,
                    now,
                    now,
                ),
            )

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if not row:
            return None
        return {
            "case_id": row["case_id"],
            "payload": json.loads(row["payload_json"]),
            "decision": row["decision"],
            "confidence": row["confidence"],
            "primary_risk_tags": json.loads(row["primary_risk_tags"] or "[]"),
            "reviewer_label": row["reviewer_label"],
        }

    def list_cases(
        self,
        decision: str | None = None,
        risk_tag: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT case_id, decision, confidence, primary_risk_tags, reviewer_label FROM cases"
        params: list[Any] = []
        clauses = []
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        if risk_tag:
            clauses.append("primary_risk_tags LIKE ?")
            params.append(f"%{risk_tag}%")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_cases(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT case_id, payload_json, decision, confidence, primary_risk_tags, reviewer_label, explanation_json FROM cases"
            ).fetchall()
        return [
            {
                "case_id": r["case_id"],
                "payload": json.loads(r["payload_json"]),
                "decision": r["decision"],
                "confidence": r["confidence"],
                "primary_risk_tags": json.loads(r["primary_risk_tags"] or "[]"),
                "reviewer_label": r["reviewer_label"],
                "explanation": json.loads(r["explanation_json"] or "{}"),
            }
            for r in rows
        ]

    def log_review_outcome(
        self,
        case_id: str,
        twin_decision: str,
        reviewer_label: str | None,
        was_correct: bool | None,
        confidence: float,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_outcomes
                (case_id, twin_decision, reviewer_label, was_correct, confidence, logged_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    twin_decision,
                    reviewer_label,
                    int(was_correct) if was_correct is not None else None,
                    confidence,
                    now,
                ),
            )
