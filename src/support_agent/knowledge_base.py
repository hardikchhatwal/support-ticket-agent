"""Chroma-backed knowledge base: chunk KB articles, embed, and retrieve.

Retrieval uses Chroma's default local embedding function (all-MiniLM-L6-v2 via
ONNX). This keeps the retrieval layer reproducible and runnable with no API key
or network after the first model download; the LLM reasoning layer is the part
that talks to OpenAI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from support_agent.config import settings
from support_agent.schema import KBHit


def _chunk(text: str, max_chars: int = 700, overlap: int = 100) -> List[str]:
    """Split article text into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                # Hard-split an oversized paragraph.
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i : i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _parse_article(path: Path) -> dict:
    """Parse a KB markdown file into id/title/category/body.

    Format: first line ``# Title``, then optional ``Category: x`` line, then body.
    The article_id is the filename stem.
    """
    raw = path.read_text(encoding="utf-8").strip()
    lines = raw.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else path.stem
    category = ""
    body_start = 1
    for i, line in enumerate(lines[1:], start=1):
        if line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()
            body_start = i + 1
            break
        if line.strip():
            break
    body = "\n".join(lines[body_start:]).strip()
    return {
        "article_id": path.stem,
        "title": title,
        "category": category,
        "body": body,
    }


class KnowledgeBase:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or settings.chroma_dir)
        self.collection_name = collection_name or settings.collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_dir(self, kb_dir: Optional[Path] = None) -> int:
        """Chunk and index every ``*.md`` article in ``kb_dir``. Returns chunk count."""
        kb_dir = Path(kb_dir or settings.kb_dir)
        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []
        for path in sorted(kb_dir.glob("*.md")):
            article = _parse_article(path)
            for idx, chunk in enumerate(_chunk(article["body"])):
                ids.append(f"{article['article_id']}::{idx}")
                docs.append(chunk)
                metas.append(
                    {
                        "article_id": article["article_id"],
                        "title": article["title"],
                        "category": article["category"],
                        "chunk_index": idx,
                    }
                )
        if not ids:
            return 0
        self.reset()
        self._collection.add(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[KBHit]:
        """Return the top-k most similar KB chunks for ``query``."""
        if self.count() == 0:
            return []
        k = top_k or settings.top_k
        result = self._collection.query(
            query_texts=[query],
            n_results=min(k, self.count()),
        )
        hits: List[KBHit] = []
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        for doc, meta, dist in zip(docs, metas, distances):
            # cosine distance -> similarity in [0, 1]
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            hits.append(
                KBHit(
                    article_id=str(meta.get("article_id", "")),
                    title=str(meta.get("title", "")),
                    text=doc,
                    score=round(score, 4),
                )
            )
        return hits
