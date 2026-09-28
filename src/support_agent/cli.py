"""Command-line interface for the triage agent.

Examples:
    support-agent ingest
    support-agent triage "I can't log in and the reset email never arrives."
    support-agent triage --file ticket.txt
    support-agent serve
"""

from __future__ import annotations

import argparse
import sys

from support_agent.config import settings
from support_agent.knowledge_base import KnowledgeBase


def _cmd_ingest(_args: argparse.Namespace) -> int:
    kb = KnowledgeBase()
    n = kb.ingest_dir(settings.kb_dir)
    print(f"Ingested {n} chunks into '{kb.collection_name}' at {kb.persist_dir}")
    return 0


def _cmd_triage(args: argparse.Namespace) -> int:
    if args.file:
        ticket_text = open(args.file, "r", encoding="utf-8").read().strip()
    elif args.text:
        ticket_text = args.text
    else:
        ticket_text = sys.stdin.read().strip()

    if not ticket_text:
        print("No ticket text provided.", file=sys.stderr)
        return 2

    if not settings.has_openai_key:
        print(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 3

    # Imported lazily so `ingest` works without the LLM stack configured.
    from support_agent.agent import TriageAgent

    kb = KnowledgeBase()
    if kb.count() == 0:
        print(
            "Knowledge base is empty. Run `support-agent ingest` first.",
            file=sys.stderr,
        )
        return 4

    decision = TriageAgent(kb=kb).triage(ticket_text)
    print(decision.model_dump_json(indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Serving on http://{args.host}:{args.port}  (API docs at /docs)")
    uvicorn.run(
        "support_agent.api:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-agent",
        description="Triage and route a single support ticket.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build the Chroma KB index.")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_triage = sub.add_parser("triage", help="Triage one ticket to a decision.")
    p_triage.add_argument("text", nargs="?", help="Ticket text (or use --file/stdin).")
    p_triage.add_argument("--file", help="Read ticket text from a file.")
    p_triage.set_defaults(func=_cmd_triage)

    p_serve = sub.add_parser("serve", help="Run the HTTP API + test page.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
