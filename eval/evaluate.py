"""Evaluate the triage agent against a labeled held-out ticket set.

Computes the quantitative metrics from the capstone spec:
  - Routing accuracy (target >= 85%)
  - Category & priority macro-F1 (target >= 0.80)
  - Escalation precision & recall (target >= 0.85 recall)
  - False auto-resolve rate (KEY SAFETY METRIC, target < 5%)

Also prints a per-ticket failure list for failure-mode analysis.

Usage:
    python eval/evaluate.py                 # uses data/tickets.jsonl
    python eval/evaluate.py --data path.jsonl --out eval/results/run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.agent import TriageAgent  # noqa: E402
from support_agent.config import settings  # noqa: E402
from support_agent.knowledge_base import KnowledgeBase  # noqa: E402


def load_tickets(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def macro_f1(gold: List[str], pred: List[str]) -> float:
    """Unweighted mean of per-class F1 over all labels seen in gold or pred."""
    labels = set(gold) | set(pred)
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def binary_precision_recall(gold: List[bool], pred: List[bool]) -> Dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def evaluate(data_path: Path, out_path: Path = None) -> dict:
    tickets = load_tickets(data_path)

    kb = KnowledgeBase()
    if kb.count() == 0:
        print("KB empty; ingesting...", file=sys.stderr)
        kb.ingest_dir(settings.kb_dir)

    agent = TriageAgent(kb=kb)

    records = []
    for t in tickets:
        decision = agent.triage(t["text"])
        records.append(
            {
                "id": t["id"],
                "text": t["text"],
                "gold": {
                    "category": t["category"],
                    "priority": t["priority"],
                    "team": t["team"],
                    "action": t["action"],
                },
                "pred": {
                    "category": decision.category.value,
                    "priority": decision.priority.value,
                    "team": decision.team.value,
                    "action": decision.action.value,
                    "kb_citation": decision.kb_citation,
                    "confidence": decision.confidence,
                },
            }
        )

    g_team = [r["gold"]["team"] for r in records]
    p_team = [r["pred"]["team"] for r in records]
    routing_accuracy = sum(1 for g, p in zip(g_team, p_team) if g == p) / len(records)

    cat_f1 = macro_f1(
        [r["gold"]["category"] for r in records],
        [r["pred"]["category"] for r in records],
    )
    pri_f1 = macro_f1(
        [r["gold"]["priority"] for r in records],
        [r["pred"]["priority"] for r in records],
    )

    # Escalation is the positive class.
    g_esc = [r["gold"]["action"] == "escalate" for r in records]
    p_esc = [r["pred"]["action"] == "escalate" for r in records]
    esc = binary_precision_recall(g_esc, p_esc)

    # False auto-resolve rate: of tickets that SHOULD escalate, fraction the
    # agent wrongly marked auto_resolve.
    should_escalate = [r for r in records if r["gold"]["action"] == "escalate"]
    false_auto = [
        r for r in should_escalate if r["pred"]["action"] == "auto_resolve"
    ]
    false_auto_rate = (
        len(false_auto) / len(should_escalate) if should_escalate else 0.0
    )

    failures = [
        {
            "id": r["id"],
            "text": r["text"],
            "gold": r["gold"],
            "pred": r["pred"],
        }
        for r in records
        if r["gold"] != {k: r["pred"][k] for k in r["gold"]}
    ]

    summary = {
        "n_tickets": len(records),
        "routing_accuracy": round(routing_accuracy, 4),
        "category_macro_f1": round(cat_f1, 4),
        "priority_macro_f1": round(pri_f1, 4),
        "escalation_precision": round(esc["precision"], 4),
        "escalation_recall": round(esc["recall"], 4),
        "false_auto_resolve_rate": round(false_auto_rate, 4),
        "targets": {
            "routing_accuracy": ">= 0.85",
            "category_macro_f1": ">= 0.80",
            "priority_macro_f1": ">= 0.80",
            "escalation_recall": ">= 0.85",
            "false_auto_resolve_rate": "< 0.05",
        },
    }

    _print_report(summary, failures)

    result = {"summary": summary, "failures": failures, "records": records}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote full results to {out_path}")
    return result


def _check(label: str, value: float, ok: bool) -> str:
    mark = "PASS" if ok else "FAIL"
    return f"  [{mark}] {label}: {value:.4f}"


def _print_report(s: dict, failures: List[dict]) -> None:
    print("\n=== Triage Agent Evaluation ===")
    print(f"Tickets evaluated: {s['n_tickets']}\n")
    print(_check("Routing accuracy", s["routing_accuracy"], s["routing_accuracy"] >= 0.85))
    print(_check("Category macro-F1", s["category_macro_f1"], s["category_macro_f1"] >= 0.80))
    print(_check("Priority macro-F1", s["priority_macro_f1"], s["priority_macro_f1"] >= 0.80))
    print(_check("Escalation precision", s["escalation_precision"], True))
    print(_check("Escalation recall", s["escalation_recall"], s["escalation_recall"] >= 0.85))
    print(
        _check(
            "False auto-resolve rate (SAFETY)",
            s["false_auto_resolve_rate"],
            s["false_auto_resolve_rate"] < 0.05,
        )
    )

    if failures:
        print(f"\n--- {len(failures)} mismatched ticket(s) for failure analysis ---")
        for f in failures:
            diffs = {
                k: (f["gold"][k], f["pred"][k])
                for k in f["gold"]
                if f["gold"][k] != f["pred"][k]
            }
            print(f"  {f['id']}: {diffs}")
    else:
        print("\nNo mismatches.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the triage agent.")
    parser.add_argument("--data", default=str(ROOT / "data" / "tickets.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "latest.json"))
    args = parser.parse_args()
    evaluate(Path(args.data), Path(args.out))


if __name__ == "__main__":
    main()
