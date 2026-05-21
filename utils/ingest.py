"""Mock dataset ingestion into long-term memory."""

from __future__ import annotations

import json
from pathlib import Path

from agents.graph import seed_memory_from_file
from tools.case_loader import load_cases_from_json
from tools.validator import validate_case_dict
from utils.config import resolve_path
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def ingest_json(path: str | Path) -> tuple[int, int]:
    """
    Load cases from JSON (list or {\"cases\": [...]}) and seed memory.
    Returns (valid_count, skipped_count).
    """
    setup_logging()
    p = resolve_path(str(path)) if not Path(path).is_absolute() else Path(path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    cases = data if isinstance(data, list) else data.get("cases", [])
    valid = 0
    skipped = 0
    for raw in cases:
        fc, err, _ = validate_case_dict(raw)
        if fc is None:
            skipped += 1
            logger.warning("Skip %s: %s", raw.get("case_id"), err)
            continue
        valid += 1
    # Reuse bulk seeder (rewrites index from file — for PoC load entire file at once)
    if str(p).endswith("seed_cases.json") or "seed" in p.name:
        seed_memory_from_file(str(p))
    else:
        # Write temp merge not needed for PoC: index individually via graph seed pattern
        from schemas.case import NormalizedCase
        from memory.long_term import LongTermMemory

        lt = LongTermMemory()
        for raw in cases:
            fc, err, _ = validate_case_dict(raw)
            if fc is None:
                continue
            norm = NormalizedCase.from_fraud_case(fc)
            label = fc.reviewer_label.value if fc.reviewer_label else None
            lt.search_tool.index_case(
                norm,
                {
                    "decision": label,
                    "primary_risk_tags": norm.primary_risk_tags,
                    "reviewer_notes": raw.get("metadata", {}).get("reviewer_notes", ""),
                },
            )
            lt.store.upsert_case(norm.case_id, raw, label, None, norm.primary_risk_tags, label)
    return valid, skipped


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/seed_cases.json"
    v, s = ingest_json(target)
    print(f"Ingested {v} cases, skipped {s}")
