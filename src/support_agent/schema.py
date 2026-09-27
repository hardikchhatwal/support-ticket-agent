"""Strongly-typed schema for the triage pipeline.

The final ``TicketDecision`` is the contract a downstream ticketing system
consumes. It is enforced with Pydantic so malformed model output fails loudly
(a validation error) rather than silently mis-routing a ticket.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    """What the ticket is fundamentally about."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    GENERAL = "general"


class Priority(str, Enum):
    """How urgently the ticket needs attention."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Team(str, Enum):
    """The destination team a ticket is routed to."""

    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    ACCOUNT_MANAGEMENT = "account_management"
    PRODUCT = "product"
    GENERAL_SUPPORT = "general_support"


class Action(str, Enum):
    """The terminal decision for a ticket."""

    AUTO_RESOLVE = "auto_resolve"
    ESCALATE = "escalate"


class TicketClassification(BaseModel):
    """Output of the structured classification LLM call (step 1)."""

    category: Category
    priority: Priority
    reasoning: str = Field(
        description="One sentence explaining the category/priority choice."
    )


class KBHit(BaseModel):
    """A single retrieved knowledge-base chunk."""

    article_id: str
    title: str
    text: str
    score: float = Field(
        description="Similarity score in [0, 1]; higher means a closer match."
    )


class RoutingDecision(BaseModel):
    """Output of the ReAct routing agent (step 3), before merging classification.

    Kept separate from ``TicketClassification`` so each LLM call has a small,
    focused output schema.
    """

    team: Team
    action: Action
    kb_citation: Optional[str] = Field(
        default=None,
        description="article_id of the KB article that justifies an auto-resolve.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the routing/resolution decision, 0-1.",
    )
    draft_resolution: Optional[str] = Field(
        default=None,
        description="Candidate reply drafted from the cited KB article (auto-resolve only).",
    )
    reasoning: str = Field(
        description="Short ReAct rationale for the team + action choice."
    )


class TicketDecision(BaseModel):
    """The single structured decision object emitted per ticket.

    Matches the design-doc contract:
    ``{category, priority, team, action, kb_citation, confidence}`` plus an
    optional drafted resolution and a human-readable rationale.
    """

    category: Category
    priority: Priority
    team: Team
    action: Action
    kb_citation: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    draft_resolution: Optional[str] = None
    reasoning: str = ""

    @classmethod
    def merge(
        cls, classification: TicketClassification, routing: RoutingDecision
    ) -> "TicketDecision":
        """Combine the two LLM outputs into the final decision object."""
        return cls(
            category=classification.category,
            priority=routing_priority_override(classification, routing),
            team=routing.team,
            action=routing.action,
            kb_citation=routing.kb_citation,
            confidence=routing.confidence,
            draft_resolution=routing.draft_resolution,
            reasoning=routing.reasoning,
        )


def routing_priority_override(
    classification: TicketClassification, routing: RoutingDecision
) -> Priority:
    """Priority is owned by the classifier; kept as a hook for future rules."""
    return classification.priority
