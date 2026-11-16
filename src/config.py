"""Load Day 7 runtime settings from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    embedding_model: str
    chroma_path: Path
    chroma_collection: str
    strategy: str
    chunks_json: Path
    questions_json: Path


def get_settings() -> Settings:
    key = os.getenv("OPENAI_API_KEY", "").strip() or "ollama"
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    strategy = os.getenv("DAY6_STRATEGY", "A").strip().upper() or "A"
    default_model = "qwen2.5" if base_url else "gpt-4o-mini"
    return Settings(
        root=ROOT,
        openai_api_key=key,
        openai_base_url=base_url,
        openai_model=os.getenv("OPENAI_MODEL", default_model).strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small").strip(),
        chroma_path=(ROOT / os.getenv("CHROMA_PATH", "day6_datas")).resolve(),
        chroma_collection=os.getenv("CHROMA_COLLECTION", f"chunks_{strategy}").strip(),
        strategy=strategy,
        chunks_json=(ROOT / "day6_datas" / "local_chunks.json").resolve(),
        questions_json=(ROOT / "day6_datas" / "questions.json").resolve(),
    )
