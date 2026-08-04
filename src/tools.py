"""
Week 2 Part 1 — tools the agent can call.

Three tools, each a plain function that returns a string for the model to
read:
  search_docs        -- the official help docs (reuses Retriever)
  search_tickets      -- past support tickets (small in-memory FAISS index)
  escalate_to_human    -- logs an escalation, optionally pings Slack

Self-test each tool directly, no LLM or agent involved:
  python -m src.tools
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache

import faiss
import requests
from langchain_core.tools import tool

from . import config
from .retrieval import Retriever, SentenceTransformerEmbedder

TICKETS_PATH = config.ROOT / "data" / "tickets.json"
ESCALATIONS_PATH = config.ROOT / "data" / "escalations.log"


# ---------------------------------------------------------------------------
# search_docs: the existing FAISS index over the help docs
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_doc_retriever() -> Retriever:
    return Retriever()


@tool
def search_docs(query: str) -> str:
    """Search the official Nimbus help documentation for how-to steps,
    policies, limits, and prices. Use this for standard product questions."""
    hits = _get_doc_retriever().search(query, k=config.TOP_K)
    if not hits:
        return "No matching documentation found."
    return "\n".join(f"[{h['source']}] {h['text']}" for h in hits)


# ---------------------------------------------------------------------------
# search_tickets: a small in-memory index over past tickets, built once.
# Not persisted to disk -- 52 tickets re-embed in well under a second, so
# there's no need for the save/load machinery retrieval.py uses for the docs.
# ---------------------------------------------------------------------------
class _TicketIndex:
    def __init__(self):
        self.tickets = json.loads(TICKETS_PATH.read_text("utf-8"))
        self.embedder = SentenceTransformerEmbedder()
        texts = [f"{t['subject']} {t['body']}" for t in self.tickets]
        vectors = self.embedder.embed_passages(texts)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.embedder.embed_query(query)
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:          # faiss pads with -1 if fewer than k exist
                continue
            hit = dict(self.tickets[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results


@lru_cache(maxsize=1)
def _get_ticket_index() -> _TicketIndex:
    return _TicketIndex()


@tool
def search_tickets(query: str) -> str:
    """Search past resolved and escalated support tickets for how similar
    real cases were handled. Use this for messy, edge-case, or "has this
    happened before" situations, or when the docs don't clearly resolve it."""
    hits = _get_ticket_index().search(query, k=3)
    if not hits:
        return "No similar past tickets found."
    return "\n".join(
        f"{h['id']} [{h['status']}] {h['subject']} -> {h['resolution']}"
        for h in hits
    )


# ---------------------------------------------------------------------------
# escalate_to_human: log the escalation, optionally notify Slack
# ---------------------------------------------------------------------------
@tool
def escalate_to_human(reason: str, question: str) -> str:
    """Escalate to a human support agent. Use when the issue needs a person:
    billing disputes, security incidents, data/privacy/legal requests, or
    when docs and past tickets do not resolve it. Provide a short reason and
    the user's question."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "reason": reason,
    }
    ESCALATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ESCALATIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if webhook:
        try:
            requests.post(
                webhook,
                json={"text": f"Escalation: {reason}\nQ: {question}"},
                timeout=5,
            )
        except Exception:
            pass  # a Slack outage should never break the escalation log

    return "Escalated to a human. A support agent will follow up."


TOOLS = [search_docs, search_tickets, escalate_to_human]


def main():
    print("search_docs:")
    print(search_docs.invoke("how much is the team plan"))

    print("\nsearch_tickets:")
    print(search_tickets.invoke("charged for someone who left the team"))

    print("\nescalate_to_human:")
    print(escalate_to_human.invoke({
        "reason": "billing dispute needs manual review",
        "question": "why was I charged for a member who left?",
    }))


if __name__ == "__main__":
    main()
