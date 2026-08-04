"""
Week 1, Day 1 — bonus taster (we go deep on this at Day 5).

Everything above was a straight line: greet -> shout. Real agents need to
*decide* where to go next. That decision lives in a CONDITIONAL EDGE: instead
of a fixed arrow, you give the graph a small function that looks at the state
and returns the name of the next step.

Here: if the question is empty, route to `refuse`; otherwise route to `answer`.

Run:  python -m src.hello_branch
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: str


def answer(state: State) -> dict:
    return {"answer": f"Answering: {state['question']}"}


def refuse(state: State) -> dict:
    return {"answer": "No question was provided, so there is nothing to answer."}


# A ROUTER is just a function: it reads state and returns a STRING — the label
# of the branch to take. It does not change state; it only picks the path.
def route(state: State) -> str:
    if state["question"].strip() == "":
        return "empty"
    return "ok"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("answer", answer)
    builder.add_node("refuse", refuse)

    # add_conditional_edges(source, router_fn, mapping)
    #   - source:     where the decision is made (here, right at START)
    #   - router_fn:  returns a label ("empty" or "ok")
    #   - mapping:    label -> which node to run next
    builder.add_conditional_edges(
        START,
        route,
        {"empty": "refuse", "ok": "answer"},
    )
    builder.add_edge("answer", END)
    builder.add_edge("refuse", END)
    return builder.compile()


def main():
    graph = build_graph()
    for q in ["How do I cancel my plan?", ""]:
        result = graph.invoke({"question": q, "answer": ""})
        print(f"input={q!r:35} -> {result['answer']}")


if __name__ == "__main__":
    main()
