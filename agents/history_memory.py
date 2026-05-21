"""Historical pass/fail context for agent prompts."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from utils.config import get_config, resolve_path


def load_review_history(limit: int = 15) -> str:
    """Summarize why prior cases passed or failed twin vs reviewer label."""
    cfg = get_config()
    db_path = resolve_path(cfg["paths"]["db_path"])
    if not db_path.exists():
        return "No prior review history indexed yet."

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT r.case_id, r.twin_decision, r.reviewer_label, r.was_correct, r.confidence,
                   c.primary_risk_tags, c.explanation_json
            FROM review_outcomes r
            LEFT JOIN cases c ON c.case_id = r.case_id
            ORDER BY r.logged_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No prior review outcomes logged yet."

    lines: list[str] = []
    for row in rows:
        correct = row["was_correct"]
        status = "PASS" if correct == 1 else "FAIL" if correct == 0 else "UNKNOWN"
        tags = row["primary_risk_tags"] or "[]"
        expl = row["explanation_json"] or "{}"
        rationale = ""
        try:
            rationale = json.loads(expl).get("analyst_rationale", "")[:120]
        except json.JSONDecodeError:
            pass
        lines.append(
            f"- {row['case_id']}: reviewer={row['reviewer_label']} twin={row['twin_decision']} "
            f"[{status}] conf={row['confidence']:.2f} tags={tags} note={rationale}"
        )
    return "Historical twin vs reviewer outcomes:\n" + "\n".join(lines)


def load_similar_decisions_from_store(limit: int = 10) -> list[dict[str, Any]]:
    cfg = get_config()
    db_path = resolve_path(cfg["paths"]["db_path"])
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT case_id, decision, reviewer_label, confidence, primary_risk_tags
            FROM cases WHERE decision IS NOT NULL
            ORDER BY updated_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
