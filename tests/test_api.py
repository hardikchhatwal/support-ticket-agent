"""HTTP API tests, fully offline: the agent is injected with TestModel."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from support_agent.agent import TriageAgent
from support_agent.api import create_app
from support_agent.knowledge_base import KnowledgeBase
from support_agent.schema import TicketDecision, TriageTrace

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "kb"


@pytest.fixture()
def client(tmp_path):
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma", collection_name="test_kb")
    kb.ingest_dir(KB_DIR)
    agent = TriageAgent(kb=kb, classifier_model=TestModel(), router_model=TestModel())
    return TestClient(create_app(agent=agent))


def test_index_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Support Ticket Triage" in res.text


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_triage_returns_valid_decision(client):
    res = client.post("/triage", json={"text": "I forgot my password."})
    assert res.status_code == 200
    TicketDecision.model_validate(res.json())


def test_triage_explain_returns_trace(client):
    res = client.post("/triage/explain", json={"text": "I forgot my password."})
    assert res.status_code == 200
    trace = TriageTrace.model_validate(res.json())
    assert trace.searches, "router should have called search_kb"


def test_kb_list_and_get(client):
    articles = client.get("/kb").json()
    ids = {a["article_id"] for a in articles}
    assert "reset_password" in ids
    one = client.get("/kb/reset_password").json()
    assert one["title"] and one["body"]
    assert client.get("/kb/does_not_exist").status_code == 404


def test_tickets_lists_labeled_set(client):
    tickets = client.get("/tickets").json()
    assert tickets and {"id", "text", "team", "action"} <= set(tickets[0])


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "   "}, {}])
def test_triage_rejects_empty_input(client, body):
    assert client.post("/triage", json=body).status_code == 422
