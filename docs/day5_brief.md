# Task brief: Week 1 Day 5 — self-correcting retrieval loop

You are working in the `support-agent` repo (a LangGraph support agent). Implement
the Day 5 feature described below. Match the existing code style exactly. Do not
break anything that already works.

## Repo context (already built, do not rewrite)
- `src/config.py` — settings read from `.env` (ANSWER_MODEL=gpt-4.1-mini, EMBED_MODEL,
  TOP_K=5). A `.env` with a real OPENAI_API_KEY already exists locally.
- `src/retrieval.py` — `Retriever().search(question, k)` returns a list of dicts:
  `{"source": str, "text": str, "score": float}`. Higher score = more relevant.
- `src/rag_graph.py` — a working LangGraph: `START -> retrieve -> answer -> END`.
  - `RagState` = TypedDict(question, chunks, answer)
  - `retrieve(state)` -> `{"chunks": hits}`  (uses `get_retriever()`)
  - `answer(state)` -> `{"answer": ...}`  (uses `get_llm()`, GPT-4.1-mini, temp 0;
    grounded system prompt; refuses if answer not in context)
  - `get_llm()` and `get_retriever()` are `@lru_cache` singletons.
  - `_format_context(chunks)` labels chunks by source.

## Environment / how to run
- Use the repo's virtualenv: `source .venv/bin/activate` before running anything.
- LangGraph version is **1.2.10**. Current API only:
  - `from langgraph.graph import StateGraph, START, END`
  - conditional routing: `builder.add_conditional_edges(source, router_fn, mapping)`
  - `set_entry_point()` is deprecated — begin with `add_edge(START, "node")`.

## Measured retrieval signal (use as defaults)
From live runs on this corpus:
- A genuinely answerable question: top chunk score ~0.65–0.70.
- An off-topic question (answer not in docs): top chunk score ~0.45 or below.
So there is a clean gap between good and bad retrieval around 0.5.

## What to build
Add a self-correcting loop so a weak retrieval gets a second (and third) chance
before the agent answers. New flow:

```
START -> retrieve -> grade_retrieval -> (if good)    answer -> END
                                      -> (if bad and attempts left) rewrite_query -> retrieve  (loop)
                                      -> (if bad and no attempts left) answer -> END   (answer node refuses gracefully)
```

### Put it in a NEW file: `src/agent_graph.py`
Leave `rag_graph.py` untouched (it's the Day 3–4 teaching artifact). The new file
should reuse the existing pieces by importing them:
`from .rag_graph import retrieve, answer, get_llm, _format_context`
(Import `get_retriever` too if needed. Reuse `answer` as-is.)

### State
Define `AgentState` (TypedDict) = the RagState fields plus:
- `attempts: int`   — how many times we've retrieved so far
- `retrieval_ok: bool` — did the last retrieval pass the grade

### Config
Add to `src/config.py`, read from env with a default:
- `MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))`  — total retrieval attempts
  allowed (original + up to 2 rewrites, matching the plan's "max 2 loops").

### Nodes
1. **retrieve (wrap the existing one)**: call the existing `retrieve`, but also
   increment the attempt counter. e.g. return `{**existing_result, "attempts": state.get("attempts", 0) + 1}`.
2. **grade_retrieval (new)**: a cheap LLM call (reuse `get_llm()`, temp 0) that
   judges whether the retrieved chunks actually contain the answer to the question.
   - Prompt it to reply with a single word: `yes` or `no`.
   - Parse robustly (strip/lower, check `startswith("yes")`).
   - Return `{"retrieval_ok": True/False}`.
   - OPTIONAL cheap guard (nice to have, not required): if the top chunk score is
     clearly high (>= 0.65) skip the LLM call and pass; if clearly low (< 0.45)
     skip and fail. Only call the LLM for the middle band. If you add thresholds,
     read them from config/env, don't hardcode magic numbers in the node.
3. **rewrite_query (new)**: an LLM call (reuse `get_llm()`, temp 0) that rewrites
   the user's question into a better search query — expand likely synonyms/product
   terms, keep it one line. Return `{"question": rewritten}`.
   - Note: rewriting the `question` field means the next `retrieve` uses the new
     query. That's intended.

### Router (conditional edge function)
`route_after_grade(state) -> str`:
- if `state["retrieval_ok"]` -> `"answer"`
- elif `state["attempts"] < config.MAX_ATTEMPTS` -> `"rewrite"`
- else -> `"answer"`   (give up looping; the grounded answer node will say it
  doesn't have the info)

### Wiring
- `add_edge(START, "retrieve")`
- `add_edge("retrieve", "grade")`
- `add_conditional_edges("grade", route_after_grade, {"answer": "answer", "rewrite": "rewrite"})`
- `add_edge("rewrite", "retrieve")`   (this is the loop)
- `add_edge("answer", END)`

Rely on the `attempts` counter to stop the loop — do NOT depend on LangGraph's
recursion limit as the safety net.

### main()
Make the loop visible when run from the CLI. Print:
- the final question actually used (may differ from input if rewritten),
- the answer,
- the sources with scores,
- the number of attempts,
- and a one-line path trace, e.g. `retrieve -> grade -> rewrite -> retrieve -> grade -> answer`.
  (Track the path however is cleanest — e.g. a list in state with an `operator.add`
  reducer, or by printing from a stream. Keep it simple.)

Signature like the existing one:
`python -m src.agent_graph "how do i stop paying"`

## Test it (run these live, they use the real model)
1. Clear hit (should pass grade on attempt 1, no rewrite):
   `python -m src.agent_graph "how much does the team plan cost?"`
   Expect: correct price, attempts = 1, path has no `rewrite`.
2. Off-topic (should loop then give up and refuse):
   `python -m src.agent_graph "what is the airspeed velocity of an unladen swallow?"`
   Expect: a refusal, attempts = MAX_ATTEMPTS, path shows `rewrite` at least once.
3. Vague-but-answerable (should ideally fail first grade, rewrite, then succeed):
   `python -m src.agent_graph "how do i stop paying"`
   Expect: ends up answering about cancelling the subscription. If it passes on
   attempt 1, that's fine too — but confirm the rewrite path works using test 2.

Print enough that a reviewer can SEE the loop happen (attempts + path trace).

## Constraints / safety
- Never print, log, or commit the contents of `.env` or the API key. `.env` is
  gitignored; keep it that way.
- Keep the teaching style of the existing files: plain functions, short comments
  that explain WHY, no clever abstractions.
- Update `README.md`: tick the Day 5 checkbox and add a one-line note that the
  agent self-corrects weak retrieval.

## When done
- Confirm all three test commands behave as described (paste their output).
- Stage the changes and make ONE commit:
  `git add . && git commit -m "Week 1 Day 5: self-correcting retrieval loop (grade + rewrite)"`
- Do NOT push and do NOT force-push. Leave pushing to me after I review.
