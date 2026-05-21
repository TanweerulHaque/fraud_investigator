"""Load case JSON from local files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.config import get_config, resolve_path


def load_case_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = resolve_path(str(path))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_cases_from_json(path: str | Path | None = None) -> list[dict[str, Any]]:
    cfg = get_config()
    p = resolve_path(path or cfg["paths"]["seed_cases"])
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("cases", [])
