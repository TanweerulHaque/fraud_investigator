"""Embedding model from config.json embeddings section."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from utils.config import get_config


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    cfg = get_config()
    emb = cfg.get("embeddings", {})
    provider = emb.get("provider")
    model = emb.get("model")
    if not provider or not model:
        raise ValueError(
            "config.json must define embeddings.provider and embeddings.model "
            "(e.g. ollama + all-minilm:22m). Run `ollama list` for installed models."
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model,
            base_url=emb.get("base_url", "http://localhost:11434"),
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)

    raise ValueError(f"Unsupported embedding provider: {provider}")


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    vec = model.embed_query(text)
    return [float(x) for x in vec]
