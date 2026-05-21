# Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Layer                               │
│  case JSON → schema validate → normalize → quality flags         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Memory & Retrieval Layer                            │
│  SQLite (cases, decisions) │ Vector similarity (past cases)      │
│  Short-term: session state   │ Long-term: embeddings + metadata  │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           Agentic Reasoning (LangGraph StateGraph)               │
│  Intake → Retrieval → Behavioral → Synthesis → Critic →          │
│  Explanation → Memory Update                                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Layer                                │
│  decision, confidence, reasons, retrieval, counterfactuals,      │
│  explanation trace, analyst rationale                              │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Observability Layer                            │
│  structured logs, optional LangSmith, evaluation artifacts       │
└─────────────────────────────────────────────────────────────────┘
```

## Layer responsibilities

| Layer | Responsibility |
|-------|----------------|
| Input | Load JSON, validate schema, normalize fields |
| Memory | Short-term graph state; long-term SQLite + vector index |
| Agents | Specialized nodes; no cross-cutting business logic |
| Output | `InvestigationResult` Pydantic artifact |
| Observability | Per-agent traces, eval metrics |

## Technology choices

- **LangGraph**: orchestration, checkpoint-friendly state
- **LangChain**: optional LLM abstraction + prompts only
- **SQLite**: durable case/decision store
- **NumPy + sklearn**: lightweight similarity (Chroma-ready interface)
- **Pydantic v2**: contracts between layers

## Extension points

- Swap `VectorStore` for Chroma/FAISS
- Replace rule-based `BehavioralPatternAgent` with feature service
- Plug production LLM gateway via `llm_factory`
- Export graph as HTTP service behind `app/` API layer
