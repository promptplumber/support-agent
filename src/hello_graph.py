"""
Week 1, Day 1 — the LangGraph mental model in one small file.

No LLM here on purpose. The point today is the machine, not the AI: how data
flows through a graph. Once this clicks, everything later (RAG, tools, memory)
is just more nodes on the same machine.

The whole model is 3 things:
  1. State  - a dict that flows through the graph. Every node reads it and
              returns a small update, which gets merged back in.
  2. Nodes  - plain Python functions. Input: the current state. Output: a dict
              with the keys it wants to change.
  3. Edges  - wiring that says which node runs next. START -> ... -> END.

Run:  python -m src.hello_graph
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
# State is the shared object that travels through the graph. We declare its
# shape with a TypedDict so we (and our editor) know what keys exist.
# Think of it as the "job ticket" that each station on the line stamps.
class HelloState(TypedDict):
    question: str        # what came in
    answer: str          # what we build up
    steps: list[str]     # a log of which nodes ran, so we can SEE the path


# ---------------------------------------------------------------------------
# 2. NODES
# ---------------------------------------------------------------------------
# A node is just a function: state in, partial-state-update out.
# It does NOT mutate state in place; it returns a dict of only the keys it
# wants to change, and LangGraph merges that back into the running state.

def greet(state: HelloState) -> dict:
    """Turn the question into a first-draft answer."""
    q = state["question"]
    draft = f"You asked: '{q}'. Here is a plain answer."
    # Return only what changed. We read the old steps list and return an
    # extended copy — that's the simple way to "append" for now. (Later, when
    # chat messages need to pile up automatically, we'll use a 'reducer' that
    # does this appending for us — but we don't need that today.)
    return {"answer": draft, "steps": state["steps"] + ["greet"]}


def shout(state: HelloState) -> dict:
    """Second station: transform the draft (here, make it loud)."""
    louder = state["answer"].upper()
    return {"answer": louder, "steps": state["steps"] + ["shout"]}


# ---------------------------------------------------------------------------
# 3. EDGES  (wiring the graph)
# ---------------------------------------------------------------------------
def build_graph():
    # StateGraph is the builder. We tell it the shape of the state it carries.
    builder = StateGraph(HelloState)

    # Register the nodes under names.
    builder.add_node("greet", greet)
    builder.add_node("shout", shout)

    # Wire the flow. START and END are built-in markers.
    # (Old tutorials use set_entry_point() — that's deprecated. An edge from
    #  START is the current way to say "begin here".)
    builder.add_edge(START, "greet")   # begin at greet
    builder.add_edge("greet", "shout") # then shout
    builder.add_edge("shout", END)     # then finish

    # compile() freezes the wiring into a runnable graph.
    return builder.compile()


def main():
    graph = build_graph()

    # invoke() runs the graph once. We hand in the STARTING state; every key
    # the graph will touch must exist here (answer/steps start empty).
    result = graph.invoke(
        {"question": "How do I reset my password?", "answer": "", "steps": []}
    )

    print("FINAL STATE")
    print("  answer:", result["answer"])
    print("  path :", " -> ".join(result["steps"]))

    # Bonus: LangGraph can draw its own wiring as a Mermaid diagram (text).
    # Paste it into https://mermaid.live to see the boxes and arrows. We'll
    # commit the rendered image into the README later.
    print("\nMERMAID DIAGRAM (paste into mermaid.live):")
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
