# Agent Design

Each agent is a **pure function** over `InvestigatorState` returning partial state updates. LangGraph merges updates.

## 1. Case Intake Agent

- **Input**: raw case dict
- **Output**: `normalized_case`, `missing_fields`, `quality_flags`
- **Logic**: Pydantic validation + field normalization (dates, amounts)

## 2. Evidence Retrieval Agent

- **Input**: normalized case
- **Tools**: similarity search, SQLite prior decisions
- **Output**: `retrieved_cases`, `matched_patterns`, `prior_reasoning_snippets`

## 3. Behavioral Pattern Agent

- **Input**: normalized case
- **Output**: `behavioral_summary`, `risk_indicators`, `stability_notes`
- **Logic**: deterministic signal extractors (geo velocity, app risk, cash-out, etc.)

## 4. Risk Synthesis Agent

- **Input**: behavioral signals + retrieval + DS scores
- **Output**: `decision`, `confidence`, `decision_drivers`
- **Logic**: weighted scorecard; optional LLM refinement when enabled

## 5. Critic / Counterfactual Agent

- **Input**: provisional decision + all evidence
- **Output**: `contradictory_evidence`, `counterfactuals`, `uncertainty_notes`
- **Logic**: challenge strong signals, surface borderline conditions

## 6. Explanation Agent

- **Input**: full state
- **Output**: `analyst_rationale`, `top_reasons`, `risk_narrative`
- **Constraint**: grounded in state fields only; no invented facts

## 7. Memory Update Agent

- **Input**: final artifacts
- **Output**: persisted case record + embedding update

## Behavioral guardrail

Agents **never** emit user-facing actions ("block account", "refund"). Language is reviewer judgment only.
