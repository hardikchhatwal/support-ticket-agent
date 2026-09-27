"""Build the Chroma index from the KB markdown corpus.

Usage:
    python scripts/ingest_kb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import settings  # noqa: E402
from support_agent.knowledge_base import KnowledgeBase  # noqa: E402


def main() -> None:
    kb = KnowledgeBase()
    n_chunks = kb.ingest_dir(settings.kb_dir)
    print(
        f"Ingested {n_chunks} chunks from {settings.kb_dir} "
        f"into collection '{kb.collection_name}' at {kb.persist_dir}"
    )


if __name__ == "__main__":
    main()
