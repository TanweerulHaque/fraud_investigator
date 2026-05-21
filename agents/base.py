"""Base agent with dual output and shared LLM utilities."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.history_memory import load_review_history
from agents.llm_factory import get_llm_for_agent
from schemas.agent_output import AgentDualOutput
from schemas.state import InvestigatorState, pack_agent_output
from tools.trace import append_trace
from utils.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

_JSON_ONLY_SUFFIX = (
    "\n\nIMPORTANT: Reply with ONE raw JSON object only. "
    "No markdown, no code fences, no prose before or after the JSON."
)


class BaseAgent:
    name: str = "base"
    system_prompt: str = (
        "You are a fraud investigator digital twin. "
        "You MUST respond with a single valid JSON object only."
    )

    def __init__(self) -> None:
        self.llm = get_llm_for_agent(self.name)

    def _history_block(self) -> str:
        return load_review_history()

    @staticmethod
    def _case_payload(state: InvestigatorState) -> str:
        case = state.get("normalized_case")
        if case:
            return case.model_dump_json()
        return json.dumps(state.get("raw_case") or {}, default=str)

    @staticmethod
    def _prior_outputs(state: InvestigatorState) -> str:
        outputs = state.get("agent_outputs") or {}
        if not outputs:
            return "{}"
        return json.dumps(outputs, default=str)[:8000]

    def invoke_json(self, user_prompt: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self.system_prompt + "\n" + self._history_block()),
            HumanMessage(content=user_prompt),
        ]
        text = self._llm_text(messages)
        try:
            return self._normalize_llm_dict(self._parse_json(text))
        except ValueError as first_err:
            logger.warning("[%s] JSON parse failed, retrying: %s", self.name, first_err)
            retry_messages = messages + [
                HumanMessage(content=user_prompt + _JSON_ONLY_SUFFIX),
            ]
            text = self._llm_text(retry_messages)
            try:
                return self._normalize_llm_dict(self._parse_json(text))
            except ValueError:
                recovered = self._recover_from_prose(text)
                if recovered:
                    logger.warning("[%s] Recovered fields from prose response", self.name)
                    return self._normalize_llm_dict(recovered)
                raise ValueError(
                    f"Agent '{self.name}' could not parse LLM JSON. Preview: {text[:400]}"
                ) from first_err

    def _llm_text(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("LLM returned empty response")

        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for obj in BaseAgent._find_json_objects(text):
            if obj:
                return obj

        raise ValueError(f"LLM did not return valid JSON. Preview: {text[:300]}")

    @staticmethod
    def _find_json_objects(text: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        found: list[dict[str, Any]] = []
        idx = 0
        while idx < len(text):
            chunk = text[idx:].lstrip()
            if not chunk or chunk[0] != "{":
                idx += 1
                continue
            try:
                obj, end = decoder.raw_decode(chunk)
                if isinstance(obj, dict):
                    found.append(obj)
                idx += end
            except json.JSONDecodeError:
                idx += 1
        return found

    @staticmethod
    def _recover_from_prose(text: str) -> dict[str, Any] | None:
        """Best-effort extraction when small models return markdown instead of JSON."""
        found = BaseAgent._find_json_objects(text)
        if found:
            return max(found, key=lambda d: len(d))

        result: dict[str, Any] = {}
        list_fields = (
            "quality_flags",
            "missing_fields",
            "contradictory_evidence",
            "counterfactuals",
            "uncertainty_notes",
            "decision_drivers",
            "top_reasons",
            "risk_indicators",
            "matched_patterns",
            "stable_signals",
            "drifting_signals",
        )
        for field in list_fields:
            val = BaseAgent._extract_inline_list(text, field)
            if val is not None:
                result[field] = val

        for field in (
            "case_summary",
            "intake_notes",
            "reasoning",
            "analyst_rationale",
            "risk_narrative",
            "summary",
            "stability_notes",
            "decision",
            "retrieval_analysis",
            "lessons_from_history",
        ):
            val = BaseAgent._extract_inline_string(text, field)
            if val:
                result[field] = val

        conf = re.search(r"confidence[\"`\s:]+([0-9.]+)", text, re.IGNORECASE)
        if conf:
            try:
                result["confidence"] = float(conf.group(1))
            except ValueError:
                pass

        return result if result else None

    @staticmethod
    def _extract_inline_list(text: str, field: str) -> list[str] | None:
        patterns = [
            rf'[`"]?{field}[`"]?\s*:\s*(\[[^\]]*\])',
            rf"\*\*\s*{field}\s*\*\*\s*:\s*(\[[^\]]*\])",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            raw = m.group(1).replace("'", '"')
            try:
                parsed = json.loads(raw)
                return BaseAgent.coerce_str_list(parsed)
            except json.JSONDecodeError:
                items = re.findall(r'"([^"]+)"', raw)
                if items:
                    return items
        return None

    @staticmethod
    def _extract_inline_string(text: str, field: str) -> str | None:
        patterns = [
            rf'[`"]?{field}[`"]?\s*:\s*"([^"]*)"',
            rf'[`"]?{field}[`"]?\s*:\s*\'([^\']*)\'',
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m and m.group(1).strip():
                return m.group(1).strip()
        return None

    @staticmethod
    def _normalize_llm_dict(raw: dict[str, Any]) -> dict[str, Any]:
        if len(raw) == 1:
            key, val = next(iter(raw.items()))
            if isinstance(val, str) and val.strip().startswith("{"):
                try:
                    inner = BaseAgent._parse_json(val)
                    if isinstance(inner, dict):
                        return inner
                except ValueError:
                    pass
            if key == "case_summary" and isinstance(val, dict):
                return val
        return raw

    @staticmethod
    def parse_risk_indicators(raw: Any) -> list:
        """Parse risk_indicators from LLM JSON into RiskIndicator models."""
        from schemas.case import RiskIndicator

        if not raw:
            return []
        items = raw if isinstance(raw, list) else [raw]
        indicators: list[RiskIndicator] = []
        for item in items:
            if isinstance(item, dict):
                try:
                    indicators.append(
                        RiskIndicator(
                            signal=str(item.get("signal") or item.get("name") or "unknown"),
                            severity=str(item.get("severity") or "medium").lower(),
                            detail=str(item.get("detail") or item.get("description") or ""),
                            weight=float(item.get("weight", 0.1)),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, str) and item.strip():
                indicators.append(
                    RiskIndicator(
                        signal=item.strip(),
                        severity="medium",
                        detail=item.strip(),
                        weight=0.1,
                    )
                )
        return indicators

    @staticmethod
    def coerce_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
                elif isinstance(item, dict):
                    parts = [f"{k}={v}" for k, v in item.items()]
                    out.append(", ".join(parts))
                elif item is not None:
                    out.append(str(item))
            return out
        if isinstance(value, str):
            s = value.strip()
            if not s or s.lower() == "none":
                return []
            if s.startswith("[") or s.startswith("{"):
                try:
                    return BaseAgent.coerce_str_list(json.loads(s))
                except json.JSONDecodeError:
                    pass
            return [s]
        return [str(value)]

    def build_dual_output(
        self,
        raw: dict[str, Any],
        structured_model: type[T],
    ) -> AgentDualOutput:
        structured = structured_model.model_validate(raw)
        return AgentDualOutput(
            agent_name=self.name,
            raw=raw,
            structured=structured.model_dump(),
        )

    def merge_state(
        self,
        state: InvestigatorState,
        updates: dict[str, Any],
        dual: AgentDualOutput,
    ) -> dict[str, Any]:
        outputs = dict(state.get("agent_outputs") or {})
        outputs[self.name] = pack_agent_output(dual)
        trace = append_trace(
            state,
            self.name,
            {"structured_keys": list(dual.structured.keys())},
        )
        return {**updates, "agent_outputs": outputs, "agent_trace": trace}
