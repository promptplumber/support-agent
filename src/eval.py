"""
Week 3 — evaluation harness.

Runs every question in data/eval_set.json through the SAME agent graph the API
serves, then scores it:

  * trajectory: did the agent take the right path (right tools / escalate / refuse)?
  * answer quality: an LLM judge compares the answer to a reference (docs + tickets only)
  * cost and latency per question

Run:  python -m src.eval
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from . import config, tools
from .rag_graph import get_llm
from .react_agent import MAX_STEPS, REFUSAL, build_graph

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "data" / "eval_set.json"
RESULTS_JSON = ROOT / "data" / "eval_results.json"   # gitignored: raw per-question rows
REPORT_MD = ROOT / "eval_report.md"                  # committed: the headline numbers

# gpt-4.1-mini list prices, USD per 1M tokens. The cost we report is an ESTIMATE.
PRICE_IN, PRICE_OUT = 0.40, 1.60

# Text the cap-escalation node (react_agent.escalate_node) answers with. That node
# calls the escalate tool directly, so it never shows up as a tool call in messages.
CAP_ESCALATION_TEXT = "passed it to a human"

JUDGE_PROMPT = """You are grading a support agent's answer against a reference.

Question: {question}
Reference (the facts a correct answer needs): {reference}
Agent's answer: {answer}

Reply with exactly one word:
correct  - the answer contains the key facts in the reference and nothing contradicts them
partial  - it has some of the key facts, or is vague/incomplete
wrong    - it is missing the key facts, contradicts the reference, or doesn't answer
"""


def judge(question: str, reference: str, answer: str) -> str:
    text = get_llm().invoke(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    ).content.lower()
    # Parse robustly: check "partial" and "wrong"/"incorrect" BEFORE "correct",
    # because "incorrect" contains the substring "correct".
    if "partial" in text:
        return "partial"
    if "wrong" in text or "incorrect" in text:
        return "wrong"
    if "correct" in text:
        return "correct"
    return "unparsed"


def run_one(graph, question: str) -> dict:
    """Run one question on a fresh thread; return answer, tools, rounds, cost, latency."""
    usage = UsageMetadataCallbackHandler()
    config_ = {
        "configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"},   # fresh thread per question
        "callbacks": [usage],
    }
    inputs = {"messages": [{"role": "user", "content": question}], "steps": 0}

    start = time.perf_counter()
    result = graph.invoke(inputs, config=config_)
    latency = time.perf_counter() - start

    # Fresh thread, so every message in the result belongs to this run.
    tools_used = [
        c["name"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for c in (m.tool_calls or [])
    ]
    # The callback sees EVERY llm call in the run, including the guardrail's.
    tokens_in = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())
    return {
        "answer": result["messages"][-1].content,
        "tools_used": tools_used,
        "tool_rounds": result.get("steps", 0),
        "latency_s": latency,
        "cost_usd": tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT,
    }


def score_trajectory(item: dict, r: dict) -> bool:
    answer = r["answer"]
    refused = REFUSAL in answer or re.search(r"(can't|cannot|unable to) help", answer, re.I) is not None
    escalated = "escalate_to_human" in r["tools_used"] or CAP_ESCALATION_TEXT in answer

    if item["must_refuse"]:
        return not r["tools_used"] and refused
    if item["must_escalate"]:
        return "escalate_to_human" in r["tools_used"]
    return (
        any(t in item["acceptable_tools"] for t in r["tools_used"])
        and not escalated
        and not refused
        and r["tool_rounds"] <= MAX_STEPS
    )


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def main():
    # Don't pollute the real escalation log (or ping Slack) with eval escalations.
    os.environ.pop("SLACK_WEBHOOK_URL", None)
    tools.ESCALATIONS_PATH = Path(tempfile.gettempdir()) / "eval_escalations.log"

    items = json.loads(EVAL_SET.read_text())
    graph = build_graph(checkpointer=MemorySaver())

    # One warmup run so model/index loading isn't counted in any question's latency.
    print("warming up...")
    run_one(graph, "How do I invite team members?")

    rows = []
    for i, item in enumerate(items, 1):
        try:
            r = run_one(graph, item["question"])
        except Exception as e:   # a crashed run is a failed question, not a crashed eval
            r = {"answer": f"ERROR: {e}", "tools_used": [], "tool_rounds": 0,
                 "latency_s": 0.0, "cost_usd": 0.0}
        r["trajectory_ok"] = score_trajectory(item, r)
        if item["bucket"] in ("docs", "tickets"):
            r["quality"] = judge(item["question"], item["reference"], r["answer"])
        else:
            r["quality"] = None
        rows.append({**item, **r})
        print(f"[{i:2}/{len(items)}] {item['id']:>3} traj={'OK ' if r['trajectory_ok'] else 'BAD'} "
              f"tools={r['tools_used']} rounds={r['tool_rounds']} q={r['quality']} {r['latency_s']:.1f}s")

    RESULTS_JSON.write_text(json.dumps(rows, indent=2))
    report = summarize(rows)
    REPORT_MD.write_text(report)
    print("\n" + report)


def summarize(rows: list[dict]) -> str:
    buckets = ["docs", "tickets", "escalate", "offtopic"]
    by = {b: [r for r in rows if r["bucket"] == b] for b in buckets}
    ok = lambda rs: sum(r["trajectory_ok"] for r in rs)

    graded = [r for r in rows if r["quality"] is not None]
    quality = {k: sum(r["quality"] == k for r in graded) for k in ("correct", "partial", "wrong", "unparsed")}
    lat = [r["latency_s"] for r in rows]
    rounds = [r["tool_rounds"] for r in rows]

    lines = [
        "# Evaluation results",
        "",
        f"- Run date: {date.today().isoformat()}",
        f"- Agent model: `{config.ANSWER_MODEL}` (judge uses the same model)",
        f"- Questions: {len(rows)} ({', '.join(f'{len(by[b])} {b}' for b in buckets)})",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Trajectory accuracy (overall) | {pct(ok(rows), len(rows))} |",
    ]
    for b in buckets:
        lines.append(f"| &nbsp;&nbsp;{b} | {pct(ok(by[b]), len(by[b]))} |")
    lines += [
        f"| Escalation correct | {ok(by['escalate'])}/{len(by['escalate'])} |",
        f"| Off-topic refusal correct | {ok(by['offtopic'])}/{len(by['offtopic'])} |",
        f"| Tool rounds (avg / max) | {np.mean(rounds):.2f} / {max(rounds)} |",
        f"| Answer quality (docs+tickets, n={len(graded)}) | "
        f"{quality['correct']} correct, {quality['partial']} partial, {quality['wrong']} wrong"
        + (f", {quality['unparsed']} unparsed" if quality["unparsed"] else "") + " |",
        f"| Avg cost / question (estimated) | ${np.mean([r['cost_usd'] for r in rows]):.5f} |",
        f"| Latency p50 / p95 (warm) | {np.percentile(lat, 50):.1f}s / {np.percentile(lat, 95):.1f}s |",
        "",
        "Notes",
        "- Cost is estimated from token counts at gpt-4.1-mini list prices "
        f"(${PRICE_IN}/1M in, ${PRICE_OUT}/1M out). It includes the guardrail call; it excludes the judge.",
        "- Each question runs on a fresh thread (no memory carry-over); one warmup run precedes timing.",
        "",
        "## Misses",
        "",
    ]
    esc_misses = [r["id"] for r in by["escalate"] if not r["trajectory_ok"]]
    if esc_misses:
        # Reported as measured; the system prompt was NOT changed to hit a target.
        lines += [
            f"**Escalation gap ({', '.join(esc_misses)}):** on these the agent answered from the docs "
            "instead of calling `escalate_to_human`. This is the soft-escalation behavior logged in "
            "`docs/known_behaviors.md` (Q2), now measured. The system prompt was not tuned for this run.",
            "",
        ]
    misses = [r for r in rows if not r["trajectory_ok"] or r["quality"] in ("wrong", "partial", "unparsed")]
    if not misses:
        lines.append("None.")
    for r in misses:
        flag = "trajectory" if not r["trajectory_ok"] else ""
        qual = f"quality={r['quality']}" if r["quality"] in ("wrong", "partial", "unparsed") else ""
        lines.append(f"- **{r['id']}** ({r['bucket']}): {r['question']} — {flag} {qual}".rstrip()
                     + f" — tools: {r['tools_used'] or 'none'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
