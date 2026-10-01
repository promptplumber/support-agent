"""
Week 2 Part 2 — the ReAct loop, built by hand.

No create_react_agent here. We build the loop ourselves so every piece is
explainable. The pattern (ReAct = Reason + Act):

    the model REASONS about the question, ACTS by calling a tool, sees the
    result, reasons again, and repeats until it can answer.

As a graph that's just two nodes in a cycle:

    agent -> (wants a tool?) -> tools -> agent -> ... -> (done) -> END
                             -> (too many rounds?) -> escalate -> END

Run:  python -m src.react_agent "I was charged for a teammate who left, what now?"
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .rag_graph import get_llm
from .tools import TOOLS, escalate_to_human
from .tracing import run_traced

# How many tool-calling rounds we allow before we stop and hand off to a human.
# This is the answer to "how do you stop it looping forever?" — a hard cap.
MAX_STEPS = int(os.getenv("MAX_TOOL_STEPS", "5"))

# Fast lookup from a tool's name to the tool itself, for the tool node.
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM = (
    "You are a support agent for a product called Nimbus. You have tools:\n"
    "- search_docs: official how-to, policies, limits, prices. Use for standard questions.\n"
    "- search_tickets: how past real cases were handled. Use for messy or edge cases,\n"
    "  or to check precedent before answering.\n"
    "- escalate_to_human: for billing disputes, security, data/privacy requests, or\n"
    "  anything you cannot confidently resolve from docs and tickets.\n"
    "Use tools before answering when the answer isn't obvious. Answer concisely and "
    "never invent prices, features, or steps."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# `messages` is the running conversation: user turns, the model's replies, the
# model's tool requests, and the tool results — all as message objects.
# The `add_messages` reducer means every node RETURNS new messages and they get
# APPENDED (not overwritten). That auto-append is exactly what a chat loop needs.
# `steps` counts how many tool rounds we've run, so we can enforce the cap.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


# ---------------------------------------------------------------------------
# The model, with tools attached
# ---------------------------------------------------------------------------
# bind_tools() tells the model which tools exist and what they're for (from each
# tool's name + docstring). The model doesn't RUN tools — it just outputs "I want
# to call search_docs with query=X". Running them is our job (the tool node).
@lru_cache(maxsize=1)
def get_model_with_tools():
    return get_llm().bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Node 1: agent — the model reasons and either asks for a tool or answers
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    # We prepend the system prompt at call time instead of storing it in state,
    # so it's always present but never duplicated in the saved history.
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]
    reply = get_model_with_tools().invoke(messages)
    # `reply` is an AIMessage. If the model wants tools, reply.tool_calls is a
    # non-empty list; if it's answering, tool_calls is empty and content is the answer.
    return {"messages": [reply]}


# ---------------------------------------------------------------------------
# Node 2: tools — run whatever the model asked for, feed results back
# ---------------------------------------------------------------------------
# (LangGraph ships a prebuilt `ToolNode` that does this. We hand-roll it so the
#  mechanism is visible: read the requested calls, run each, return a ToolMessage
#  per call tagged with its tool_call_id so the model can match result to request.)
def tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]        # the AIMessage that requested tools
    results = []
    for call in last.tool_calls:        # each: {"name", "args", "id"}
        tool = TOOLS_BY_NAME[call["name"]]
        output = tool.invoke(call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    # Append the results AND count this as one completed tool round.
    return {"messages": results, "steps": state.get("steps", 0) + 1}


# ---------------------------------------------------------------------------
# The conditional edge: this function IS the loop's brain
# ---------------------------------------------------------------------------
# It looks at the model's last message and decides the next step. Three outcomes:
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "end"                    # model gave a final answer -> done
    if state.get("steps", 0) >= MAX_STEPS:
        return "escalate"               # too many rounds -> hand to a human
    return "tools"                      # model wants a tool -> go run it


# ---------------------------------------------------------------------------
# Node 3: escalate — the safety net when we hit the cap
# ---------------------------------------------------------------------------
# The plan's rule: on hitting the cap, escalate instead of guessing. Cap alone
# is not enough — a cap with a graceful fallback is the real answer.
def escalate_node(state: AgentState) -> dict:
    question = _first_user_text(state)
    escalate_to_human.invoke(
        {"reason": "reached the tool-call limit without resolving", "question": question}
    )
    msg = AIMessage(
        content="I couldn't resolve this confidently, so I've passed it to a "
                "human support agent who will follow up."
    )
    return {"messages": [msg]}


def _first_user_text(state: AgentState) -> str:
    for m in state["messages"]:
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


# ---------------------------------------------------------------------------
# Wiring the loop
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("escalate", escalate_node)

    builder.add_edge(START, "agent")
    # After the agent speaks, should_continue picks the route:
    builder.add_conditional_edges(
        "agent", should_continue,
        {"tools": "tools", "escalate": "escalate", "end": END},
    )
    builder.add_edge("tools", "agent")     # <- the loop: results go back to the model
    builder.add_edge("escalate", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# CLI: run a question, show the answer and which tools were used
# ---------------------------------------------------------------------------
def main():
    # Optional --thread for memory across turns (same wiring as Part 1).
    thread = "demo"
    args = sys.argv[1:]
    if "--thread" in args:
        i = args.index("--thread")
        thread = args[i + 1]
        args = args[:i] + args[i + 2:]
    question = " ".join(args) or "I was charged for a teammate who left, what now?"

    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string("data/memory.sqlite") as saver:
        graph = build_graph(checkpointer=saver)
        # Pass steps=0 each turn so the cap is PER QUESTION, not per conversation.
        inputs = {"messages": [{"role": "user", "content": question}], "steps": 0}
        config = {"configurable": {"thread_id": thread}}
        # Remember how many messages the thread already holds so we can report
        # only THIS turn's tool calls (the checkpointer returns full history).
        prior_state = graph.get_state(config)
        prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0
        result, trace_id = run_traced(
            graph, inputs, trace_name="react-agent",
            tags=["week2", "react"], metadata={"question": question}, config=config,
        )

    # Show the final answer and the tools the model actually used this turn.
    final = result["messages"][-1].content
    new_messages = result["messages"][prior_count:]
    tools_used = [
        c["name"]
        for m in new_messages
        if isinstance(m, AIMessage)
        for c in (m.tool_calls or [])
    ]
    print("Q:", question)
    print("\nA:", final)
    print("\nTools used:", tools_used or "(none)")
    print("Tool rounds:", result.get("steps", 0))
    if trace_id:
        print("Trace:", trace_id)


if __name__ == "__main__":
    main()
