"""LLM factory — one model per agent from config.json."""

from __future__ import annotations

import os
from typing import Any

from utils.config import REQUIRED_LLM_AGENTS, get_config


def get_llm_for_agent(agent_name: str) -> Any:
    if agent_name not in REQUIRED_LLM_AGENTS:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Expected one of: {list(REQUIRED_LLM_AGENTS)}"
        )

    cfg = get_config()
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("use_llm", True):
        raise RuntimeError(
            "LLM is disabled in config.json (llm.use_llm=false). "
            "Enable it for agentic reasoning."
        )

    agents_cfg = llm_cfg.get("agents") or {}
    model = agents_cfg.get(agent_name)
    if not model or not str(model).strip():
        raise ValueError(
            f"No LLM model configured for agent '{agent_name}'. "
            f"Set llm.agents.{agent_name} in config.json to an Ollama model "
            f"(run `ollama list` to see installed models). "
            f"Currently configured: {list(agents_cfg.keys())}"
        )

    provider = llm_cfg.get("provider", "ollama")
    base_url = llm_cfg.get("base_url", "http://localhost:11434")
    temperature = llm_cfg.get("temperature", 0.1)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            format="json",
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def setup_langsmith() -> None:
    cfg = get_config()
    ls = cfg.get("langsmith", {})
    if ls.get("enabled") and os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = ls.get("project", "fraud-investigator-twin")
    else:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
