# Fraud Investigator Twin

Local PoC of a **digital twin** that imitates a fraud reviewer’s judgment style: inspect structured evidence, retrieve similar cases via **ChromaDB**, synthesize a binary decision (`fraud` | `not_fraud`), critique its reasoning with LLM agents, and produce grounded explainability.

This is **not** a recommendation engine or generic chatbot. It does not suggest customer actions.

## Design choices (summary)

| Area | Choice |
|------|--------|
| Orchestration | LangGraph linear `StateGraph` over `InvestigatorState` |
| Core reasoning | LLM agents only (required; `llm.use_llm: true`) |
| Embeddings | Ollama/OpenAI model from `config.json` → `embeddings` |
| Memory | SQLite + Chroma (handcrafted features + summary embedding) |
| Contracts | Pydantic v2 models |
| Tracing | Structured logs; LangSmith optional |

See `docs/` for problem statement, architecture, agent/memory/tool/eval/explainability plans.

## Project structure

```
fraud_investigator/
├── agents/          # LangGraph nodes + graph
├── app/             # Streamlit UI
├── data/            # Mock seed + eval cases
├── docs/            # Design documents
├── evaluation/      # Metrics + harness
├── memory/          # Short/long-term + vector store
├── prompts/         # LLM prompt templates (optional)
├── schemas/         # Pydantic + JSON schema
├── tools/           # Validators, SQLite, similarity, etc.
├── utils/           # Config + logging
├── config.json
├── main.py          # CLI
└── requirements.txt
```

## Setup

```bash
cd fraud_investigator
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` for LangSmith.

## Configuration

Edit `config.json`:

- `llm.use_llm` — `false` (default) uses deterministic synthesis/explanation
- `llm.provider` — `ollama` or `openai`
- `llm.agents` — per-agent model names
- `decision.*` — fraud / review thresholds
- `langsmith.enabled` — optional tracing

## Run

### Seed historical memory

```bash
python main.py seed
```

### Investigate one case

```bash
python main.py investigate data/cases/sample_case.json
python main.py investigate data/cases/sample_case.json -o data/last_result.json
python main.py demo
```

### Evaluation harness

```bash
python main.py evaluate
python main.py evaluate --verbose
```

Writes `data/eval_report.json` with accuracy, F1, FPR/FNR, explanation proxy, retrieval relevance, consistency.

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Pipeline (LangGraph)

1. **Case Intake** — validate + LLM quality assessment  
2. **Embedding** — LLM case summary + embedding vector  
3. **Evidence Retrieval** — Chroma similarity + LLM interpretation  
4. **Behavioral Pattern** — LLM risk signals  
5. **Risk Synthesis** — LLM `fraud` / `not_fraud` judgment  
6. **Critic** — LLM counterfactuals  
7. **Explanation** — LLM analyst rationale  
8. **Memory Update** — SQLite + Chroma upsert  

Each agent file under `agents/` publishes **raw** (for other agents) and **structured** (downstream) outputs in `state.agent_outputs`.

```bash
python -m main graph   # print/save LangGraph artifact
```

## Output artifact

Each run returns `InvestigationResult` with structured `ExplainabilityArtifact`:

- Decision, confidence, top signals  
- Similar cases, counterfactuals, uncertainty  
- Analyst rationale + agent trace  

## Mock data

- `data/seed_cases.json` — seeds long-term memory (5 cases)  
- `data/eval_cases.json` — labeled eval set (good / fraud / borderline)  
- `data/cases/sample_case.json` — CLI demo case  

## Enabling LLM (optional)

1. Start Ollama locally and pull models referenced in `config.json`  
2. Set `"use_llm": true`  
3. Re-run investigate/evaluate  

## Extending to production

Replace components independently:

- `memory/vector_store.py` → Chroma/FAISS service  
- `agents/nodes.py` behavioral rules → feature platform  
- `tools/sqlite_store.py` → warehouse API  
- `agents/graph.py` → HTTP worker with checkpoints  

## License

Internal PoC — adjust as needed.
