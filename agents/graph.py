"""LangGraph orchestration for Fraud Investigator Twin."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agents.behavioral_pattern_agent import run_behavioral_pattern
from agents.case_intake_agent import run_case_intake
from agents.critic_agent import run_critic
from agents.embedding_agent import run_embedding
from agents.evidence_retrieval_agent import run_evidence_retrieval
from agents.explanation_agent import run_explanation
from agents.graph_visualization import export_graph_artifact
from agents.llm_factory import setup_langsmith
from agents.memory_update_agent import run_memory_update
from agents.risk_synthesis_agent import run_risk_synthesis
from memory.long_term import LongTermMemory
from schemas.case import ReviewerDecision
from schemas.output import InvestigationResult
from schemas.state import InvestigatorState
from tools.explanation_formatter import build_explainability_artifact
from utils.config import reload_config
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_long_term: LongTermMemory | None = None
_compiled_graph = None


def get_long_term_memory() -> LongTermMemory:
    global _long_term
    if _long_term is None:
        _long_term = LongTermMemory()
    return _long_term


def build_investigator_graph(
    long_term: LongTermMemory | None = None,
    export_graph: bool = True,
):
    global _compiled_graph
    lt = long_term or get_long_term_memory()

    graph = StateGraph(InvestigatorState)
    graph.add_node("case_intake", lambda s: run_case_intake(s))
    graph.add_node("embedding", lambda s: run_embedding(s))
    graph.add_node("evidence_retrieval", lambda s: run_evidence_retrieval(s, lt))
    graph.add_node("behavioral_pattern", lambda s: run_behavioral_pattern(s))
    graph.add_node("risk_synthesis", lambda s: run_risk_synthesis(s))
    graph.add_node("critic", lambda s: run_critic(s))
    graph.add_node("explanation", lambda s: run_explanation(s))
    graph.add_node("memory_update", lambda s: run_memory_update(s, lt))

    graph.set_entry_point("case_intake")
    graph.add_edge("case_intake", "embedding")
    graph.add_edge("embedding", "evidence_retrieval")
    graph.add_edge("evidence_retrieval", "behavioral_pattern")
    graph.add_edge("behavioral_pattern", "risk_synthesis")
    graph.add_edge("risk_synthesis", "critic")
    graph.add_edge("critic", "explanation")
    graph.add_edge("explanation", "memory_update")
    graph.add_edge("memory_update", END)

    compiled = graph.compile()
    _compiled_graph = compiled

    if export_graph:
        export_graph_artifact(compiled)

    return compiled


def run_investigation(
    raw_case: dict[str, Any],
    long_term: LongTermMemory | None = None,
    export_graph: bool = False,
) -> InvestigationResult:
    reload_config()
    setup_logging()
    setup_langsmith()

    if "reviewer_label" in raw_case:
        del raw_case["reviewer_label"]

    initial: InvestigatorState = {
        "raw_case": raw_case,
        "agent_trace": [],
        "agent_outputs": {},
        "missing_fields": [],
        "quality_flags": [],
        "retrieved_cases": [],
        "matched_patterns": [],
        "prior_reasoning_snippets": [],
        "decision_drivers": [],
    }

    app = build_investigator_graph(long_term, export_graph=export_graph)
    logger.info("Running investigation...")
    final = app.invoke(initial)

    if final.get("error"):
        raise ValueError(final["error"])

    decision = final.get("decision") or ReviewerDecision.NOT_FRAUD
    confidence = final.get("confidence") or 0.0
    case = final.get("normalized_case")
    case_id = case.case_id if case else raw_case.get("case_id", "unknown")

    logger.info("Building explainability artifact...")
    explainability = build_explainability_artifact(
        case_id=case_id,
        decision=decision,
        confidence=confidence,
        decision_drivers=final.get("decision_drivers") or [],
        behavioral=final.get("behavioral_summary"),
        retrieved=final.get("retrieved_cases") or [],
        critic=final.get("critic_output"),
        explanation=final.get("explanation_output"),
        agent_trace=final.get("agent_trace") or [],
    )

    logger.info("Returning investigation result...")
    return InvestigationResult(
        case_id=case_id,
        decision=decision,
        confidence=confidence,
        decision_drivers=final.get("decision_drivers") or [],
        behavioral_summary=final.get("behavioral_summary"),
        retrieved_cases=final.get("retrieved_cases") or [],
        critic=final.get("critic_output"),
        explanation=final.get("explanation_output"),
        explainability=explainability,
        agent_trace=final.get("agent_trace") or [],
        agent_outputs=final.get("agent_outputs") or {},
        quality_flags=final.get("quality_flags") or [],
        missing_fields=final.get("missing_fields") or [],
    )


def seed_memory_from_file(path: str | None = None, force: bool = False) -> int:
    """Seed Chroma + SQLite using embedding agent (summary + vector)."""
    from agents.embedding_agent import EmbeddingAgent
    from tools.case_loader import load_cases_from_json
    from tools.validator import validate_case_dict
    from schemas.case import NormalizedCase

    lt = get_long_term_memory()

    if not force:
        existing = lt.store.list_cases(limit=1)
        if existing:
            logger.info("Memory already seeded, skipping.")
            return 0

    cases = load_cases_from_json(path)
    embed_agent = EmbeddingAgent()
    count = 0

    for index, raw in enumerate(cases):
        logger.info("-"* 25)
        logger.info(f"Processing case {index + 1} of {len(cases)}")
        logger.info("Validating case...")
        fc, err, _ = validate_case_dict(raw)
        if fc is None:
            logger.warning("Skip invalid seed case: %s", err)
            continue
        logger.info("Normalizing case...")
        norm = NormalizedCase.from_fraud_case(fc)
        state: InvestigatorState = {"raw_case": raw, "normalized_case": norm, "agent_outputs": {}}
        logger.info("Embedding case...")
        result = embed_agent.run(state)
        if result.get("error"):
            logger.warning("Skip embed %s: %s", norm.case_id, result["error"])
            continue

        summary = result.get("case_summary", "")
        vector = result.get("case_embedding", [])
        label = fc.reviewer_label.value if fc.reviewer_label else None
        expl = raw.get("metadata", {}).get("reviewer_notes", "")

        logger.info("Remembering investigation...")
        lt.remember_investigation(
            case=norm,
            payload=raw,
            case_summary=summary,
            query_vector=vector,
            decision=label or "",
            confidence=0.0,
            explanation={"analyst_rationale": expl},
        )
        count += 1

    logger.info("Seeded %d cases into Chroma + SQLite", count)
    return count
