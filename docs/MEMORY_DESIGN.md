# Memory Design

## Short-term memory

Held in LangGraph `InvestigatorState` for the active investigation:

- Raw and normalized case
- Per-agent intermediate outputs
- Active reasoning trace (`agent_trace`)

Cleared when a new case run starts.

## Long-term memory

Persisted locally:

| Store | Contents |
|-------|----------|
| SQLite `cases` | case_id, payload JSON, decision, confidence, timestamps |
| SQLite `reviewer_labels` | ground truth / human labels for eval |
| Vector index | embedding per case for similarity retrieval |

## Retrieval modes

1. **Similarity**: cosine on engineered feature vector
2. **By risk type**: filter metadata `primary_risk_tag`
3. **By reviewer decision**: SQL filter on `decision`
4. **By outcome correctness**: eval feedback column `was_correct`

## Memory update flow

After each investigation, Memory Update Agent:

1. Inserts/updates case row
2. Refreshes embedding in vector store
3. Logs review outcome for harness

## Future production mapping

- Short-term → Redis session / graph checkpoint
- Long-term → warehouse + vector DB service
- Embeddings → managed embedding API
