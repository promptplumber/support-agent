"""
Week 2 Part 1 — a working ReAct agent using LangGraph's prebuilt loop.

We hand-build the ReAct loop ourselves as a separate learning exercise; this
file's job is just to prove the tools + memory work end to end.
`create_react_agent` gives us the loop (call model -> maybe call tools ->
repeat until no more tool calls) for free.

Ask a question:
  python -m src.react_prebuilt "how much does the team plan cost?"

Same --thread across calls = the agent remembers the conversation:
  python -m src.react_prebuilt --thread demo1 "how do I invite people?"
  python -m src.react_prebuilt --thread demo1 "what did I just ask you?"
"""

from __future__ import annotations

import argparse

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from . import config
from .rag_graph import get_llm
from .tools import TOOLS
from .tracing import run_traced

MEMORY_PATH = config.ROOT / "data" / "memory.sqlite"
DEFAULT_THREAD = "cli-demo"

SYSTEM_PROMPT = (
    "You are a support agent for a product called Nimbus. Answer concisely "
    "and never invent facts. Choose tools like this:\n"
    "- search_docs: standard how-to, policy, limit, or pricing questions.\n"
    "- search_tickets: messy or edge-case situations, or 'has this happened "
    "before' questions -- use this when the docs alone don't clearly "
    "resolve it.\n"
    "- escalate_to_human: issues that need a person -- billing disputes, "
    "security incidents, data/privacy/legal requests, or anything docs and "
    "tickets don't resolve.\n"
    "Look things up with search_docs and/or search_tickets before answering "
    "a product question.\n"
    "HARD RULE: if the user is asking for an EXCEPTION to a policy the "
    "docs state plainly -- a refund outside the refund window, a limit "
    "they want waived, a fee they want reversed -- that is a billing "
    "dispute. Do not just quote the policy and refuse. You MUST call "
    "search_tickets for precedent and then call escalate_to_human; a "
    "policy exception is always a human's call, never yours.\n"
    "Escalate rather than guess."
)


def _tool_calls_made(messages: list) -> list[str]:
    """Pull the name of every tool the agent decided to call, in order."""
    names = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            names.append(call["name"])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--thread", default=DEFAULT_THREAD)
    args = parser.parse_args()

    thread_config = {"configurable": {"thread_id": args.thread}}
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(MEMORY_PATH)) as saver:
        agent = create_react_agent(
            model=get_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT, checkpointer=saver
        )
        # invoke() returns the WHOLE thread's message history, not just this
        # turn's -- remember how many messages existed before, so we can
        # report tool calls for this turn only, not every past turn too.
        prior_state = agent.get_state(thread_config)
        prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

        result, trace_id = run_traced(
            agent,
            {"messages": [{"role": "user", "content": args.question}]},
            trace_name="react-prebuilt",
            session_id=args.thread,
            tags=["week2", "react-prebuilt"],
            metadata={"input_question": args.question, "model": config.ANSWER_MODEL},
            config=thread_config,
        )

    new_messages = result["messages"][prior_count:]
    answer = new_messages[-1].content
    tool_calls = _tool_calls_made(new_messages)

    print("Q:", args.question)
    print("Thread:", args.thread)
    print("\nA:", answer)
    print("\nTools called:", " -> ".join(tool_calls) if tool_calls else "(none)")
    if trace_id:
        print(f"Trace: {trace_id}  (open it in your Langfuse dashboard)")


if __name__ == "__main__":
    main()
