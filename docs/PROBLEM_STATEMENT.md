# Problem Statement

## Fraud Investigator Digital Twin

Financial platforms must review user cases for fraud using heterogeneous evidence: profiles, device signals, transaction history, app usage, location patterns, model scores, and prior human decisions. Expert fraud reviewers apply tacit judgment—weighting weak signals, comparing to past cases, and articulating rationale—rather than issuing product recommendations.

## Goal

Build a **local PoC digital twin** that imitates a fraud-reviewer's decision style:

1. Consume structured case evidence
2. Retrieve similar historical cases
3. Summarize suspicious vs normal behavioral patterns
4. Synthesize a fraud judgment (`fraud` | `not_fraud` | `needs_review`)
5. Critique its own reasoning (counterfactuals, uncertainty)
6. Produce a human-readable, evidence-grounded explanation
7. Store the case and outcome for future memory

## Non-goals

- Not a recommendation engine (no "you should block/refund/contact user")
- Not a generic chatbot or tutor
- Not production-scale infra (mock data, local stores)

## Success criteria (PoC)

- Modular pipeline replaceable component-by-component
- Every decision includes structured explainability
- Evaluation harness measures agreement with reviewer labels on mock cases
- Runnable locally with minimal setup
