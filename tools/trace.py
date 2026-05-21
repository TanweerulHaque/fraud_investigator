"""Append agent step to investigation trace."""

from __future__ import annotations

from typing import Any

from schemas.state import InvestigatorState


def append_trace(state: InvestigatorState, agent: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(state.get("agent_trace") or [])
    trace.append({"agent": agent, **summary})
    return trace
