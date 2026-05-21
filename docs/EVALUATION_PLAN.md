# Evaluation Plan

## Objectives

Answer:

1. Does the twin match reviewer ground-truth labels?
2. Does it cite relevant evidence (proxy: overlap with labeled risk tags)?
3. Are borderline cases routed to `needs_review`?
4. Are explanations structurally complete?

## Metrics (implemented in `evaluation/harness.py`)

| Metric | Description |
|--------|-------------|
| Accuracy | Decision == label |
| Precision / Recall / F1 | Per-class and macro for `fraud` |
| FPR / FNR | False positive/negative rates on fraud class |
| Confidence calibration | Brier-style binning (simplified) |
| Explanation proxy | Required fields present + reason count |
| Retrieval relevance | Similar cases share label (top-k) |
| Case consistency | Same case_id re-run yields same decision |

## Test sets

- `data/seed_cases.json` — seeds long-term memory
- `data/eval_cases.json` — held-out labeled cases (good / fraud / borderline)

## Running

```bash
python main.py evaluate
python main.py evaluate --verbose
```

## Human review extension (future)

- LLM-as-judge on explanation groundedness
- Blind A/B vs human analyst notes
