from pathlib import Path

import pytest

from support_agent.knowledge_base import KnowledgeBase, _chunk, _parse_article

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "kb"


def test_chunker_respects_size_and_is_nonempty():
    text = "\n\n".join(f"Paragraph number {i} with some words." for i in range(20))
    chunks = _chunk(text, max_chars=200, overlap=20)
    assert chunks
    assert all(len(c) <= 200 for c in chunks)


def test_parse_article_extracts_title_and_category():
    article = _parse_article(KB_DIR / "reset_password.md")
    assert article["article_id"] == "reset_password"
    assert "password" in article["title"].lower()
    assert article["category"] == "account"
    assert article["body"]


@pytest.fixture()
def kb(tmp_path):
    store = KnowledgeBase(persist_dir=tmp_path / "chroma", collection_name="test_kb")
    store.ingest_dir(KB_DIR)
    return store


def test_ingest_populates_collection(kb):
    assert kb.count() > 0


def test_retrieve_returns_relevant_article(kb):
    hits = kb.retrieve("I forgot my password and need to reset it", top_k=3)
    assert hits
    assert hits[0].article_id == "reset_password"
    assert 0.0 <= hits[0].score <= 1.0


def test_retrieve_billing_query(kb):
    hits = kb.retrieve("how do I get a refund for a recent charge", top_k=3)
    ids = {h.article_id for h in hits}
    assert "refund_policy" in ids
