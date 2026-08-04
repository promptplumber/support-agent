"""
Week 1, Day 3-4 — the RAG graph.

Same LangGraph machine as the hello-world, but the two nodes now do real work:
  retrieve : find the most relevant doc chunks for the question
  answer   : GPT-4.1-mini writes an answer using ONLY those chunks

Ask a question:
  python -m src.rag_graph "how much does the team plan cost?"
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from . import config
from .retrieval import Retriever


# ---------------------------------------------------------------------------
# State: what flows through this graph
# ---------------------------------------------------------------------------
class RagState(TypedDict):
    question: str
    chunks: list[dict]   # what retrieval found (source, text, score)
    answer: str          # what the model wrote


# ---------------------------------------------------------------------------
# Lazy singletons: build the retriever and model once, reuse across calls.
# (Loading the embedder + index is slow; we don't want it per request.)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


@lru_cache(maxsize=1)
def get_llm():
    from langchain_openai import ChatOpenAI

    config.require_openai_key()
    # temperature=0 -> as deterministic as possible. For support answers we want
    # the safe, consistent response every time, not creative variety.
    return ChatOpenAI(model=config.ANSWER_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def retrieve(state: RagState) -> dict:
    """Node 1: pull the top-k chunks for the question."""
    hits = get_retriever().search(state["question"], k=config.TOP_K)
    return {"chunks": hits}


SYSTEM_PROMPT = (
    "You are a support agent for a product called Nimbus. "
    "Answer the user's question using ONLY the context provided. "
    "If the answer is not in the context, say you don't have that information "
    "and suggest contacting support. Keep answers short and direct. "
    "Do not invent features, prices, or steps that are not in the context."
)


def _format_context(chunks: list[dict]) -> str:
    # Label each chunk with its source doc so the model can ground its answer.
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)


def answer(state: RagState) -> dict:
    """Node 2: write an answer grounded in the retrieved chunks."""
    context = _format_context(state["chunks"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Context:\n{context}\n\nQuestion: {state['question']}"},
    ]
    reply = get_llm().invoke(messages)
    return {"answer": reply.content}


# ---------------------------------------------------------------------------
# Graph wiring: START -> retrieve -> answer -> END
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def main():
    question = " ".join(sys.argv[1:]) or "How do I reset my password?"
    graph = build_graph()
    result = graph.invoke({"question": question, "chunks": [], "answer": ""})

    print("Q:", question)
    print("\nA:", result["answer"])
    print("\nSources used:")
    for c in result["chunks"]:
        print(f"  - {c['source']}  (score {c['score']:.2f})")


if __name__ == "__main__":
    main()
