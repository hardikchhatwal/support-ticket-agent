# Support Ticket Triage & Routing Agent

A ReAct agent over a RAG knowledge base that turns **one** incoming support
ticket into **one** strongly-typed routing decision.

Given a single ticket, the agent makes three bounded decisions:

1. **Classify** the ticket by `category` and `priority` (structured LLM call).
2. **Route** it to the correct destination `team`.
3. **Resolve or escalate** — decide whether the ticket can be auto-resolved from
   the knowledge base (with a cited article + drafted reply) or must be
   escalated to a human.

The output is a single validated decision object:

```json
{
  "category": "account",
  "priority": "high",
  "team": "account_management",
  "action": "auto_resolve",
  "kb_citation": "reset_password",
  "confidence": 0.92,
  "draft_resolution": "You can reset your password yourself: on the sign-in page click 'Forgot password'...",
  "reasoning": "The reset_password article answers this ticket directly."
}
```

Scope is deliberately narrow: one ticket at a time, routing decisions only. It
does **not** hold multi-turn conversations, generate open-ended replies beyond a
retrieved KB resolution, or manage ticket lifecycle state.

---

## Architecture

```
                ┌─────────────────────────────────────────────────────┐
   ticket text  │  1. Classifier (Pydantic AI, structured output)      │
 ──────────────▶│     → TicketClassification{category, priority}       │
                └───────────────────────────┬─────────────────────────┘
                                            │ category + priority
                                            ▼
                ┌─────────────────────────────────────────────────────┐
                │  2/3. Router (Pydantic AI ReAct loop)                │
                │       tool: search_kb ──▶ Chroma vector store        │
                │       reasons over hits → RoutingDecision            │
                └───────────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                ┌─────────────────────────────────────────────────────┐
                │  Safety guardrail: downgrade auto_resolve→escalate   │
                │  unless cited AND confidence ≥ threshold             │
                └───────────────────────────┬─────────────────────────┘
                                            ▼
                                     TicketDecision (validated)
```

### Framework choices

| Concern | Choice | Why |
|---|---|---|
| Agent + schema enforcement | **Pydantic AI** | Runs the ReAct tool-calling loop *and* enforces the output schema. A downstream ticketing system consumes these fields, so malformed output must fail loudly (a validation error) rather than silently mis-route. |
| KB retrieval | **Chroma** | Low setup overhead, simple local persistence at capstone scale. |
| ReAct loop | **Pydantic AI tool-calling** | The control flow is linear (retrieve → reason → decide). Pydantic AI's native tool loop *is* the ReAct loop, so a second agent framework (smolagents was evaluated in the design doc) would be unjustified complexity. LangGraph was likewise ruled out — no graph orchestration or state branching is needed. |

Retrieval uses Chroma's default local embedding model (`all-MiniLM-L6-v2` via
ONNX), so the RAG layer is reproducible and needs no API key. Only the
LLM reasoning layer (classification + ReAct) calls OpenAI.

---

## Project layout

```
.
├── src/support_agent/
│   ├── schema.py          # Pydantic decision schema + enums (the contract)
│   ├── config.py          # env-driven settings
│   ├── knowledge_base.py  # Chroma ingest / chunk / retrieve
│   ├── agent.py           # classifier + ReAct router + safety guardrail
│   └── cli.py             # `support-agent ingest` / `triage`
├── data/
│   ├── kb/                # knowledge-base corpus (markdown articles)
│   └── tickets.jsonl      # labeled held-out ticket set (ground truth)
├── eval/
│   └── evaluate.py        # spec metrics + failure-mode listing
├── scripts/ingest_kb.py   # one-shot KB indexer
└── tests/                 # offline tests (Pydantic AI TestModel/FunctionModel)
```

---

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # then add your OPENAI_API_KEY
```

---

## Usage

**1. Build the knowledge-base index** (once, or whenever KB articles change):

```bash
support-agent ingest
# or: python scripts/ingest_kb.py
```

**2. Triage a ticket:**

```bash
support-agent triage "I can't log in and the password reset email never arrives."
support-agent triage --file ticket.txt
echo "How do I cancel my subscription?" | support-agent triage
```

The command prints the `TicketDecision` as JSON.

---

## Evaluation

The spec defines four quantitative metrics against a labeled held-out set
(`data/tickets.jsonl`):

```bash
python eval/evaluate.py
```

| Metric | Target |
|---|---|
| Routing accuracy | ≥ 85% |
| Category / priority macro-F1 | ≥ 0.80 |
| Escalation recall | ≥ 0.85 |
| **False auto-resolve rate** (key safety metric) | **< 5%** |

The harness also prints every mismatched ticket for **failure-mode analysis**
(ambiguous tickets, multi-issue tickets, etc.) and writes a full JSON report to
`eval/results/latest.json`.

**On the safety metric.** A wrong auto-resolve (telling a customer a problem is
solved when it isn't) is far costlier than an unnecessary escalation. Two
mechanisms defend it: (1) the router is prompted to escalate when in doubt, and
(2) a deterministic guardrail in `agent.py` downgrades any `auto_resolve` that
lacks a KB citation or falls below the confidence threshold.

### Dataset & reproducibility

`data/tickets.jsonl` is a small curated, fully labeled ticket set with a matching
KB corpus in `data/kb/`, so the project is self-contained and reproducible with
no proprietary or personal data. The evaluator reads any JSONL with the fields
`{id, text, category, priority, team, action}`, so a larger public
support-routing dataset (Kaggle / Hugging Face) can be dropped in via
`python eval/evaluate.py --data path/to/dataset.jsonl`.

---

## Testing

```bash
pytest
```

Tests run fully offline: the LLM is replaced with Pydantic AI's
`TestModel`/`FunctionModel`, while Chroma retrieval runs for real. Covered:
schema validation, KB chunking/ingest/retrieval, the safety guardrail, and the
end-to-end ReAct pipeline wiring.
