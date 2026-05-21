"""Short-term session memory wrapper."""

from __future__ import annotations

from schemas.state import InvestigatorState


class ShortTermMemory:
    """Holds active investigation context (LangGraph state)."""

    def __init__(self) -> None:
        self._state: InvestigatorState = {}

    def reset(self, raw_case: dict) -> InvestigatorState:
        self._state = {
            "raw_case": raw_case,
            "agent_trace": [],
            "missing_fields": [],
            "quality_flags": [],
            "retrieved_cases": [],
            "matched_patterns": [],
            "prior_reasoning_snippets": [],
            "decision_drivers": [],
        }
        return self._state

    @property
    def state(self) -> InvestigatorState:
        return self._state

    def update(self, patch: InvestigatorState) -> None:
        self._state.update(patch)
