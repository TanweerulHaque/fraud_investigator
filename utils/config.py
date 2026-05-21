"""Load and validate config.json from project root."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"

# Must match `name` on each BaseAgent subclass in agents/
REQUIRED_LLM_AGENTS = (
    "case_intake",
    "embedding",
    "evidence_retrieval",
    "behavioral_pattern",
    "risk_synthesis",
    "critic",
    "explanation",
    "memory_update",
)


def validate_config(data: dict[str, Any]) -> None:
    llm = data.get("llm", {})
    if llm.get("use_llm", True):
        agents = llm.get("agents") or {}
        missing = [a for a in REQUIRED_LLM_AGENTS if not agents.get(a)]
        if missing:
            raise ValueError(
                "config.json llm.agents is missing models for: "
                + ", ".join(missing)
                + f". Required keys: {list(REQUIRED_LLM_AGENTS)}"
            )
        empty = [a for a in REQUIRED_LLM_AGENTS if not str(agents.get(a, "")).strip()]
        if empty:
            raise ValueError(
                "config.json llm.agents has empty model names for: " + ", ".join(empty)
            )
        if not llm.get("provider"):
            raise ValueError("config.json llm.provider is required")

    emb = data.get("embeddings", {})
    if not emb.get("model"):
        raise ValueError("config.json embeddings.model is required")
    if not emb.get("provider"):
        raise ValueError("config.json embeddings.provider is required")


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    validate_config(data)
    return data


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    return load_config()


def reload_config() -> dict[str, Any]:
    """Clear cache after editing config.json (e.g. in a long-running shell)."""
    get_config.cache_clear()
    return get_config()


def resolve_path(relative: str) -> Path:
    return PROJECT_ROOT / relative
