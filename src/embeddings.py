"""Sentence-Transformers E5 embeddings compatible with Day 6 index."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_queries(texts: list[str], model_name: str) -> list[list[float]]:
    model = get_model(model_name)
    prefixed = [t if t.lower().startswith("query:") else f"query: {t}" for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str, model_name: str) -> list[float]:
    return embed_queries([text], model_name)[0]
