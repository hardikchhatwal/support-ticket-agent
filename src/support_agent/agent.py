"""The triage pipeline: structured classification + a ReAct routing agent.

Design (per the capstone spec):
  1. Classify the ticket's category and priority with a structured LLM call.
  2. Retrieve candidate KB articles via embedding search (a ReAct tool).
  3. Reason (ReAct) over the retrieved articles to route the ticket and decide
     auto-resolve vs. escalate, emitting a strongly-typed decision object.

Pydantic AI provides both the ReAct tool-calling loop and output-schema
enforcement, so a separate agent framework (smolagents was evaluated in the
design doc) is unnecessary for this linear control flow.

A deterministic safety guardrail runs after the agent: an ``auto_resolve`` is
downgraded to ``escalate`` unless it cites a KB article AND clears the
confidence threshold. This directly targets the spec's key safety metric,
false auto-resolve rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic_ai import Agent, RunContext

from support_agent.config import settings
from support_agent.knowledge_base import KnowledgeBase
from support_agent.schema import (
    Action,
    KBHit,
    RoutingDecision,
    TicketClassification,
    TicketDecision,
)

CLASSIFY_PROMPT = """You are a support-ticket classifier.
Given one support ticket, assign exactly one category and one priority.

Categories:
- billing: payments, invoices, refunds, plans, subscriptions.
- technical: bugs, crashes, errors, API issues, data export.
- account: sign-in, passwords, 2FA, profile, account recovery.
- feature_request: asking for new functionality that does not exist.
- general: anything else, greetings, unclear asks.

Priority:
- urgent: outage, data loss, security issue, or blocked business-critical work.
- high: broken core functionality for the user, failed payment, cannot sign in.
- medium: degraded experience with a workaround.
- low: questions, cosmetic issues, feature requests.

Return the category, priority, and one sentence of reasoning."""

ROUTE_PROMPT = """You are a support-ticket routing agent.
You receive a ticket plus its category and priority. Decide (a) which team to
route it to and (b) whether it can be auto-resolved from the knowledge base or
must be escalated to a human.

ALWAYS call `search_kb` at least once with a focused query before deciding.

Teams:
- billing: payments, invoices, refunds, plans, subscriptions.
- technical_support: bugs, crashes, errors, API, data export.
- account_management: sign-in, passwords, 2FA, account recovery.
- product: feature requests and product feedback.
- general_support: unclear or general questions.

Decision rules:
- auto_resolve ONLY when a retrieved KB article directly and completely answers
  the ticket. Set kb_citation to that article's id and write a short, specific
  draft_resolution grounded in that article. Set confidence to how well the
  article matches (be honest; 0.8+ means a near-exact match).
- escalate when no article matches, the match is partial, the ticket is
  ambiguous or multi-issue, or it clearly needs human judgement (billing
  disputes, account recovery without backup codes, persistent bugs, urgent
  incidents). Leave kb_citation null and draft_resolution null.

When in doubt, escalate. A wrong auto-resolve is far more costly than an
unnecessary escalation. Always explain your reasoning briefly."""


@dataclass
class TriageDeps:
    """Dependencies injected into the routing agent's tools."""

    kb: KnowledgeBase


def build_classifier(model: Optional[object] = None) -> Agent:
    """Structured classification agent (no tools)."""
    return Agent(
        model or settings.model,
        output_type=TicketClassification,
        system_prompt=CLASSIFY_PROMPT,
    )


def build_router(model: Optional[object] = None) -> Agent:
    """ReAct routing agent with a KB search tool."""
    agent = Agent(
        model or settings.model,
        deps_type=TriageDeps,
        output_type=RoutingDecision,
        system_prompt=ROUTE_PROMPT,
    )

    @agent.tool
    def search_kb(ctx: RunContext[TriageDeps], query: str) -> List[dict]:
        """Search the knowledge base for articles relevant to the ticket.

        Returns a list of {article_id, title, score, text} ordered by relevance.
        """
        hits: List[KBHit] = ctx.deps.kb.retrieve(query, top_k=settings.top_k)
        return [h.model_dump() for h in hits]

    return agent


def _apply_safety_guardrail(
    routing: RoutingDecision, threshold: float
) -> RoutingDecision:
    """Force an escalation unless an auto-resolve is well-cited and confident."""
    if routing.action == Action.AUTO_RESOLVE:
        if routing.kb_citation is None or routing.confidence < threshold:
            return routing.model_copy(
                update={
                    "action": Action.ESCALATE,
                    "draft_resolution": None,
                    "reasoning": (
                        f"{routing.reasoning} [Guardrail: downgraded to escalate "
                        f"(confidence {routing.confidence:.2f} < "
                        f"{threshold:.2f} or no citation)]"
                    ),
                }
            )
    return routing


class TriageAgent:
    """End-to-end triage pipeline wrapping the two agents and the KB."""

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        classifier_model: Optional[object] = None,
        router_model: Optional[object] = None,
        autoresolve_threshold: Optional[float] = None,
    ) -> None:
        self.kb = kb or KnowledgeBase()
        self.classifier = build_classifier(classifier_model)
        self.router = build_router(router_model)
        self.threshold = (
            autoresolve_threshold
            if autoresolve_threshold is not None
            else settings.autoresolve_threshold
        )

    def triage(self, ticket_text: str) -> TicketDecision:
        classification = self.classifier.run_sync(ticket_text).output

        route_input = (
            f"Ticket:\n{ticket_text}\n\n"
            f"Classified category: {classification.category.value}\n"
            f"Classified priority: {classification.priority.value}"
        )
        routing = self.router.run_sync(
            route_input, deps=TriageDeps(kb=self.kb)
        ).output
        routing = _apply_safety_guardrail(routing, self.threshold)

        return TicketDecision.merge(classification, routing)
