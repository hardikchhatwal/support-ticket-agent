"""Agent tests that run fully offline using Pydantic AI's TestModel/FunctionModel.

No OpenAI key or network is required: the LLM is replaced with a local test
model, while the Chroma retrieval layer runs for real against the KB corpus.
"""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from support_agent.agent import TriageAgent, _apply_safety_guardrail
from support_agent.knowledge_base import KnowledgeBase
from support_agent.schema import Action, RoutingDecision, Team

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "kb"


@pytest.fixture()
def kb(tmp_path):
    store = KnowledgeBase(persist_dir=tmp_path / "chroma", collection_name="test_kb")
    store.ingest_dir(KB_DIR)
    return store


def test_guardrail_downgrades_uncited_auto_resolve():
    routing = RoutingDecision(
        team=Team.BILLING,
        action=Action.AUTO_RESOLVE,
        kb_citation=None,
        confidence=0.95,
        reasoning="looks resolvable",
    )
    out = _apply_safety_guardrail(routing, threshold=0.75)
    assert out.action == Action.ESCALATE
    assert out.draft_resolution is None


def test_guardrail_downgrades_low_confidence_auto_resolve():
    routing = RoutingDecision(
        team=Team.BILLING,
        action=Action.AUTO_RESOLVE,
        kb_citation="refund_policy",
        confidence=0.4,
        reasoning="weak match",
    )
    out = _apply_safety_guardrail(routing, threshold=0.75)
    assert out.action == Action.ESCALATE


def test_guardrail_keeps_valid_auto_resolve():
    routing = RoutingDecision(
        team=Team.BILLING,
        action=Action.AUTO_RESOLVE,
        kb_citation="refund_policy",
        confidence=0.9,
        reasoning="strong match",
    )
    out = _apply_safety_guardrail(routing, threshold=0.75)
    assert out.action == Action.AUTO_RESOLVE


def test_pipeline_end_to_end_with_test_model(kb):
    """TestModel auto-generates schema-valid outputs and exercises the tool."""
    agent = TriageAgent(
        kb=kb,
        classifier_model=TestModel(),
        router_model=TestModel(),
    )
    decision = agent.triage("I forgot my password and can't log in.")
    assert decision.category is not None
    assert decision.team is not None
    assert decision.action in (Action.AUTO_RESOLVE, Action.ESCALATE)
    assert 0.0 <= decision.confidence <= 1.0


def test_router_uses_kb_and_can_auto_resolve(kb):
    """A scripted router model retrieves, then returns a confident cited resolve."""
    calls = {"searched": False}

    def classify_fn(messages, info: AgentInfo):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "category": "account",
                        "priority": "high",
                        "reasoning": "sign-in problem",
                    },
                )
            ]
        )

    def route_fn(messages, info: AgentInfo):
        if not calls["searched"]:
            calls["searched"] = True
            return ModelResponse(
                parts=[ToolCallPart(tool_name="search_kb", args={"query": "reset password"})]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "team": "account_management",
                        "action": "auto_resolve",
                        "kb_citation": "reset_password",
                        "confidence": 0.92,
                        "draft_resolution": "Use Forgot password to reset it.",
                        "reasoning": "reset_password article answers this exactly",
                    },
                )
            ]
        )

    agent = TriageAgent(
        kb=kb,
        classifier_model=FunctionModel(classify_fn),
        router_model=FunctionModel(route_fn),
    )
    trace = agent.triage_explained("I forgot my password and can't log in.")
    decision = trace.decision
    assert calls["searched"] is True
    assert decision.action == Action.AUTO_RESOLVE
    assert decision.kb_citation == "reset_password"
    assert decision.team == Team.ACCOUNT_MANAGEMENT
    assert trace.classification_reasoning == "sign-in problem"
    assert not trace.guardrail_applied
    assert [s.query for s in trace.searches] == ["reset password"]
    assert trace.searches[0].hits[0].article_id == "reset_password"
