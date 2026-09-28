"""HTTP API and test page for the triage agent.

Endpoints:
    GET  /                minimal HTML page for triaging tickets by hand
    GET  /health          liveness + KB readiness
    POST /triage          {"text": "..."} -> TicketDecision
    POST /triage/explain  {"text": "..."} -> TriageTrace (decision + intermediate steps)
    GET  /kb              list KB articles
    GET  /kb/{id}         one KB article
    GET  /tickets         the labeled ticket set (for side-by-side testing)

Run with ``support-agent serve`` (defaults to http://127.0.0.1:8000).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from support_agent.config import PROJECT_ROOT, settings
from support_agent.knowledge_base import KnowledgeBase, _parse_article
from support_agent.schema import TicketDecision, TriageTrace

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"
TICKETS_PATH = PROJECT_ROOT / "data" / "tickets.jsonl"


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000, description="Ticket text.")


class KBArticle(BaseModel):
    article_id: str
    title: str
    category: str
    body: str


class LabeledTicket(BaseModel):
    id: str
    text: str
    category: str
    priority: str
    team: str
    action: str


def _load_articles() -> List[KBArticle]:
    return [
        KBArticle(**_parse_article(p)) for p in sorted(Path(settings.kb_dir).glob("*.md"))
    ]


def create_app(agent: Optional[object] = None) -> FastAPI:
    """Build the app. Pass ``agent`` to inject a pre-built (e.g. test) TriageAgent."""
    app = FastAPI(
        title="Support Ticket Triage Agent",
        description="Triage one support ticket into a structured routing decision.",
        version="0.1.0",
    )
    state = {"agent": agent}

    def get_agent():
        # Built lazily so the server starts (and /health works) without a key.
        if state["agent"] is None:
            if not settings.has_openai_key:
                raise HTTPException(
                    503, "OPENAI_API_KEY is not set. Add it to .env and restart."
                )
            from support_agent.agent import TriageAgent

            kb = KnowledgeBase()
            if kb.count() == 0:
                raise HTTPException(
                    503, "Knowledge base is empty. Run `support-agent ingest` first."
                )
            state["agent"] = TriageAgent(kb=kb)
        return state["agent"]

    def run_triage(text: str) -> TriageTrace:
        text = text.strip()
        if not text:
            raise HTTPException(422, "Ticket text is empty.")
        try:
            return get_agent().triage_explained(text)
        except HTTPException:
            raise
        except Exception as exc:  # surface model/API failures as a clean 502
            raise HTTPException(502, f"Triage failed: {type(exc).__name__}: {exc}")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": settings.model,
            "openai_key_set": settings.has_openai_key,
            "autoresolve_threshold": settings.autoresolve_threshold,
            "kb_articles": len(list(Path(settings.kb_dir).glob("*.md"))),
            "kb_chunks": KnowledgeBase().count(),
        }

    # Sync handlers: FastAPI runs them in a worker thread, so the blocking
    # run_sync calls inside TriageAgent don't stall the event loop.
    @app.post("/triage", response_model=TicketDecision)
    def triage(req: TriageRequest) -> TicketDecision:
        return run_triage(req.text).decision

    @app.post("/triage/explain", response_model=TriageTrace)
    def triage_explain(req: TriageRequest) -> TriageTrace:
        return run_triage(req.text)

    @app.get("/kb", response_model=List[KBArticle])
    def list_articles() -> List[KBArticle]:
        return _load_articles()

    @app.get("/kb/{article_id}", response_model=KBArticle)
    def get_article(article_id: str) -> KBArticle:
        for article in _load_articles():
            if article.article_id == article_id:
                return article
        raise HTTPException(404, f"No KB article '{article_id}'.")

    @app.get("/tickets", response_model=List[LabeledTicket])
    def list_tickets() -> List[LabeledTicket]:
        if not TICKETS_PATH.exists():
            return []
        lines = TICKETS_PATH.read_text(encoding="utf-8").splitlines()
        return [LabeledTicket(**json.loads(line)) for line in lines if line.strip()]

    return app


app = create_app()
