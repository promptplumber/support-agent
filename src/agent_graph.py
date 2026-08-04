"""
Week 1, Day 5 — self-correcting retrieval loop.

The Day 3-4 graph always answers after a single retrieval, even when that
retrieval missed. This graph adds a grade step: if the retrieved chunks don't
actually contain the answer, rewrite the question into a better search query
and retrieve again (up to config.MAX_ATTEMPTS times) before giving up and
letting the grounded answer node refuse gracefully.

New flow:
  START -> retrieve -> grade -> (good)                  answer -> END
                              -> (bad, attempts left)    rewrite -> retrieve (loop)
                              -> (bad, no attempts left) answer -> END

Ask a question:
  python -m src.agent_graph "how do i stop paying"
"""

from __future__ import annotations

import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from . import config
from .rag_graph import answer, get_llm, get_retriever, _format_context
from .tracing import run_traced, tracing_enabled


# ---------------------------------------------------------------------------
# State: the RagState fields plus the loop's own bookkeeping
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str
    attempts: int          # how many times we've retrieved so far
    retrieval_ok: bool     # did the last retrieval pass the grade
    path: Annotated[list[str], operator.add]   # nodes visited, for the CLI trace


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def retrieve(state: AgentState) -> dict:
    """Node: same retrieval as Day 3-4, plus the attempt counter."""
    hits = get_retriever().search(state["question"], k=config.TOP_K)
    return {
        "chunks": hits,
        "attempts": state.get("attempts", 0) + 1,
        "path": ["retrieve"],
    }


GRADE_PROMPT = (
    "You are checking retrieval quality for a support agent. Given a question "
    "and the context retrieved for it, decide whether the context actually "
    "contains the answer to the question. Reply with a single word: yes or no."
)


def grade_retrieval(state: AgentState) -> dict:
    """Node: does the retrieved context actually answer the question?

    Cheap guard first: a top chunk score that's clearly high or clearly low
    skips the LLM call entirely (config.GRADE_HIGH_SCORE / GRADE_LOW_SCORE).
    Only the ambiguous middle band pays for a model call.
    """
    top_score = state["chunks"][0]["score"] if state["chunks"] else 0.0

    if top_score >= config.GRADE_HIGH_SCORE:
        return {"retrieval_ok": True, "path": ["grade"]}
    if top_score < config.GRADE_LOW_SCORE:
        return {"retrieval_ok": False, "path": ["grade"]}

    context = _format_context(state["chunks"])
    messages = [
        {"role": "system", "content": GRADE_PROMPT},
        {"role": "user",
         "content": f"Question: {state['question']}\n\nContext:\n{context}"},
    ]
    reply = get_llm().invoke(messages).content.strip().lower()
    return {"retrieval_ok": reply.startswith("yes"), "path": ["grade"]}


REWRITE_PROMPT = (
    "You rewrite vague support questions into better search queries. Expand "
    "likely synonyms and product terms so the query matches documentation "
    "more directly. Reply with the rewritten query only, as a single line."
)


def rewrite_query(state: AgentState) -> dict:
    """Node: turn a weak query into a better one before retrying retrieval."""
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": state["question"]},
    ]
    reply = get_llm().invoke(messages).content.strip()
    return {"question": reply, "path": ["rewrite"]}


# ---------------------------------------------------------------------------
# Router: reads state, returns the label of the branch to take
# ---------------------------------------------------------------------------
def route_after_grade(state: AgentState) -> str:
    if state["retrieval_ok"]:
        return "answer"
    if state["attempts"] < config.MAX_ATTEMPTS:
        return "rewrite"
    return "answer"  # out of attempts -- let the grounded answer node refuse


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade_retrieval)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("answer", answer)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"answer": "answer", "rewrite": "rewrite"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("answer", END)
    return builder.compile()


def main():
    question = " ".join(sys.argv[1:]) or "How do I reset my password?"
    graph = build_graph()
    result, trace_id = run_traced(
        graph,
        {
            "question": question,
            "chunks": [],
            "answer": "",
            "attempts": 0,
            "retrieval_ok": False,
            "path": [],
        },
        tags=["week1", "agent"],
        metadata={"input_question": question, "model": config.ANSWER_MODEL},
    )

    print("Q (original):", question)
    print("Q (final):   ", result["question"])
    print("\nA:", result["answer"])
    print("\nSources used:")
    for c in result["chunks"]:
        print(f"  - {c['source']}  (score {c['score']:.2f})")
    print(f"\nAttempts: {result['attempts']}")
    # "answer" is reused as-is from rag_graph.py, so it never appends to the
    # path itself -- but it's always the node that ends the graph, so we can
    # append it here for a trace that matches what actually ran.
    print("Path:", " -> ".join(result["path"] + ["answer"]))
    if trace_id:
        print(f"\nTrace: {trace_id}  (open it in your Langfuse dashboard)")
    elif not tracing_enabled():
        print("\nTracing disabled (set LANGFUSE_* in .env)")


if __name__ == "__main__":
    main()
