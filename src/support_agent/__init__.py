"""Support Ticket Triage & Routing Agent.

A ReAct agent (Pydantic AI) over a RAG knowledge base (Chroma) that turns a
single incoming support ticket into one strongly-typed routing decision.
"""

from support_agent.schema import (
    Action,
    Category,
    Priority,
    Team,
    TicketClassification,
    TicketDecision,
)

__all__ = [
    "Action",
    "Category",
    "Priority",
    "Team",
    "TicketClassification",
    "TicketDecision",
]

__version__ = "0.1.0"
