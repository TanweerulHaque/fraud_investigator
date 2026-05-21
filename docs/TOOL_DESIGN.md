# Tool Design

PoC tools are thin wrappers callable from agents or injected as dependencies.

| Tool | Purpose |
|------|---------|
| `validate_case_schema` | JSON Schema + Pydantic validation |
| `load_case_file` | Read case JSON from `data/cases/` |
| `similarity_search` | Top-k similar cases from vector store |
| `sqlite_read_cases` | Query historical cases / decisions |
| `sqlite_write_case` | Persist investigation outcome |
| `embedding_store` | Build/query case embeddings |
| `format_explanation` | Structured explanation artifact |
| `trace_log` | Agent-step debug logging |
| `mock_ds_score_lookup` | Return canned model score context |

No extraneous tools (email, ticketing, payment APIs).

## Design principle

Tools return **structured dicts**, not natural language. Agents interpret results.
