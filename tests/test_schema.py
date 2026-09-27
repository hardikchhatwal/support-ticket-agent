import pytest
from pydantic import ValidationError

from support_agent.schema import (
    Action,
    Category,
    Priority,
    RoutingDecision,
    Team,
    TicketClassification,
    TicketDecision,
)


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        RoutingDecision(
            team=Team.BILLING,
            action=Action.ESCALATE,
            confidence=1.5,
            reasoning="x",
        )


def test_invalid_enum_fails_loudly():
    with pytest.raises(ValidationError):
        TicketClassification(category="nonsense", priority="low", reasoning="x")


def test_merge_combines_classification_and_routing():
    classification = TicketClassification(
        category=Category.BILLING, priority=Priority.HIGH, reasoning="payment issue"
    )
    routing = RoutingDecision(
        team=Team.BILLING,
        action=Action.AUTO_RESOLVE,
        kb_citation="refund_policy",
        confidence=0.9,
        draft_resolution="Here is how to get a refund...",
        reasoning="matched refund article",
    )
    decision = TicketDecision.merge(classification, routing)
    assert decision.category == Category.BILLING
    assert decision.priority == Priority.HIGH
    assert decision.team == Team.BILLING
    assert decision.action == Action.AUTO_RESOLVE
    assert decision.kb_citation == "refund_policy"
    assert decision.confidence == 0.9
