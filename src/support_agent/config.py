"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("SUPPORT_AGENT_MODEL", "openai:gpt-4o-mini")
    autoresolve_threshold: float = float(
        os.getenv("SUPPORT_AGENT_AUTORESOLVE_THRESHOLD", "0.75")
    )
    top_k: int = int(os.getenv("SUPPORT_AGENT_TOP_K", "4"))
    chroma_dir: Path = Path(
        os.getenv("SUPPORT_AGENT_CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma"))
    )
    collection_name: str = os.getenv("SUPPORT_AGENT_COLLECTION", "kb_articles")
    kb_dir: Path = PROJECT_ROOT / "data" / "kb"

    @property
    def has_openai_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


settings = Settings()
