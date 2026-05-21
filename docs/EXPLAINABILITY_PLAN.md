# Explainability Plan

Every investigation produces an `InvestigationResult` with:

## Structured sections

1. **Decision** — `fraud` | `not_fraud` | `needs_review`
2. **Confidence** — 0–1 with calibration notes
3. **Top evidence signals** — ranked behavioral + model drivers
4. **Similar past cases** — case_id, similarity, prior decision
5. **Counterfactuals** — "If X were false, decision might shift to Y"
6. **Uncertainty notes** — from critic agent
7. **Analyst rationale** — prose summary grounded in above

## Question templates answered

- What signals drove the decision?
- Which past cases were similar?
- What evidence would change the outcome?
- Why is confidence high or low?

## Grounding rules

- Explanation agent may only reference fields present in state
- Template fallback when LLM disabled
- `explanation_trace` lists agent contributions for audit

## Output formats

- JSON artifact (`result.json` per run)
- CLI rich panel
- Streamlit expandable sections
