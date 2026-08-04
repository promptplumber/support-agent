# Task brief: Week 2 Part 1 — tools, ticket search, short-term memory

Work in the `support-agent` repo. Build the three tools, get a WORKING agent using
LangGraph's **prebuilt** ReAct agent, and add short-term (per-conversation) memory.

IMPORTANT SCOPE LIMIT: Use the prebuilt `create_react_agent` ONLY. Do **not**
hand-build the ReAct loop with custom nodes — that is being done separately as a
learning exercise. Your job is the tools + a working prebuilt agent + memory.

## Repo context (already built — do not modify these)
- `src/config.py` — settings from `.env` (ANSWER_MODEL=gpt-4.1-mini, TOP_K, paths).
- `src/retrieval.py` — `Retriever().search(q, k)` -> `[{source,text,score}]`; and
  `SentenceTransformerEmbedder` (reuse this to embed tickets — don't add a new model).
- `src/rag_graph.py` — has `get_llm()` (cached ChatOpenAI, gpt-4.1-mini, temp 0). Reuse it.
- `src/tracing.py` — `run_traced(graph, inputs, *, trace_name, session_id, tags, metadata)`
  returns `(result, trace_id)`, degrades gracefully. Reuse it for the agent run.
- `data/tickets.json` — 52 fake past tickets. Each: `{id, subject, body, category, status, resolution}`.
  `status` is `resolved` or `escalated`.

## Verified current API (use exactly these — do not use older variants)
- Tools: `from langchain_core.tools import tool`. The function's **docstring becomes the
  tool description the model uses to choose** — write these carefully.
- Prebuilt agent: `from langgraph.prebuilt import create_react_agent`.
  Signature has `model, tools, prompt, checkpointer`. Pass a model instance (reuse
  `get_llm()`), `tools=[...]`, `prompt="<system prompt>"`, and `checkpointer=saver`.
- Memory: `from langgraph.checkpoint.sqlite import SqliteSaver`.
  `SqliteSaver.from_conn_string(path)` is a **context manager**:
  ```python
  with SqliteSaver.from_conn_string("data/memory.sqlite") as saver:
      agent = create_react_agent(model=get_llm(), tools=TOOLS, prompt=SYSTEM, checkpointer=saver)
      agent.invoke({"messages": [{"role": "user", "content": q}]},
                   config={"configurable": {"thread_id": thread_id}})
  ```
  Same `thread_id` across turns = the agent remembers the conversation.
  (NOTE: the `with` form is fine for this CLI. The FastAPI server later needs a
  different, longer-lived wiring — leave that for later, don't solve it now.)

## Build

### 1. `src/tools.py` — three tools
Each is a plain function decorated with `@tool`, returning a **string** (the model
reads the string). Docstrings must be clear and DISTINCT so the model doesn't confuse
search_docs with search_tickets.

- **`search_docs(query: str) -> str`**
  Docstring idea: "Search the official Nimbus help documentation for how-to steps,
  policies, limits, and prices. Use this for standard product questions."
  Impl: call the existing `Retriever().search(query, k=TOP_K)`, format the top hits as
  `[source] text` lines. Build the Retriever lazily once (module-level cache), like retrieval.py.

- **`search_tickets(query: str) -> str`**
  Docstring idea: "Search past resolved and escalated support tickets for how similar
  real cases were handled. Use this for messy, edge-case, or 'has this happened before'
  situations, or when the docs don't clearly resolve it."
  Impl: load `data/tickets.json` once. Embed each ticket's `subject + " " + body` using
  `SentenceTransformerEmbedder` (reuse it), normalize, and do cosine search with a small
  in-memory `faiss.IndexFlatIP` (same pattern as retrieval.py — no need to persist to disk).
  Return top 3 as lines like: `T-1011 [escalated] Refund after 20 days -> <resolution>`.
  Cache the embedded index in memory so it isn't rebuilt every call.

- **`escalate_to_human(reason: str, question: str) -> str`**
  Docstring idea: "Escalate to a human support agent. Use when the issue needs a person:
  billing disputes, security incidents, data/privacy/legal requests, or when docs and
  past tickets do not resolve it. Provide a short reason and the user's question."
  Impl: append one JSON line to `data/escalations.log` with `{timestamp, question, reason}`.
  If env var `SLACK_WEBHOOK_URL` is set, also POST a short message there (use `requests`;
  wrap in try/except so a webhook failure never crashes the tool). Return a confirmation
  string like: "Escalated to a human. A support agent will follow up." Must work with no
  webhook configured.

Add `TOOLS = [search_docs, search_tickets, escalate_to_human]`.

### 2. Tool self-test (in `tools.py` `__main__`, or `scripts/test_tools.py`)
Call each tool directly with a sample input and print the result, so we can confirm the
tools work BEFORE putting an agent in front of them. Samples:
- `search_docs("how much is the team plan")`
- `search_tickets("charged for someone who left the team")`
- `escalate_to_human("billing dispute needs manual review", "why was I charged for a member who left?")`

### 3. `src/react_prebuilt.py` — the working prebuilt agent + memory
- Build a system prompt that tells the agent HOW to choose tools: docs for standard
  how-to/policy/price questions; tickets for messy/edge cases or precedent; escalate for
  human-needed issues or when docs+tickets don't resolve it. Tell it to answer concisely
  and not invent facts.
- Compile `create_react_agent(model=get_llm(), tools=TOOLS, prompt=SYSTEM, checkpointer=saver)`
  inside the `SqliteSaver.from_conn_string(...)` context manager.
- CLI: accept a question and an optional `--thread` id (default a fixed demo id). Run the
  agent via `run_traced(...)` so it's traced. Print the final answer AND which tools were
  called (inspect the message history for tool calls) so tool choice is visible.

## Test it (run live)
1. **Tool choice — docs:** `python -m src.react_prebuilt "how much does the team plan cost?"`
   Expect: it calls `search_docs`, answers ~$9/member/month.
2. **Tool choice — tickets:** `python -m src.react_prebuilt "I was charged for a teammate who already left, what now?"`
   Expect: it calls `search_tickets`, references how that case was handled (T-1008).
3. **Tool choice — escalate:** `python -m src.react_prebuilt "I want a refund but it's been 20 days"`
   Expect: it calls `escalate_to_human` (past ticket T-1011 shows this is outside the 14-day
   window and was escalated). A line should appear in `data/escalations.log`.
4. **Memory (short-term):** two turns, same thread:
   ```
   python -m src.react_prebuilt --thread demo1 "how do I invite people?"
   python -m src.react_prebuilt --thread demo1 "what did I just ask you?"
   ```
   Expect: the second answer correctly recalls the first question. Then run the second
   command with a DIFFERENT `--thread` and confirm it does NOT remember (proves isolation).

Print enough that tool choice and memory are visibly demonstrated.

## Constraints / safety
- Do NOT hand-roll the ReAct loop. Prebuilt agent only. (Hand-roll is a separate session.)
- Reuse existing pieces (`Retriever`, `SentenceTransformerEmbedder`, `get_llm`, `run_traced`).
  Don't add a second embedding model or a second LLM config path.
- Never print or commit secrets. `.env`, `data/memory.sqlite`, `data/escalations.log`
  should be gitignored (add them to `.gitignore`).
- Keep the teaching style: plain functions, short comments explaining WHY.
- Update `README.md`: tick the Week 2 tools/memory items and add a one-line note.

## When done
- Paste the output of all four tests (showing tool choices + the memory recall).
- One commit: `git add . && git commit -m "Week 2 Part 1: tools (docs/tickets/escalate) + prebuilt ReAct agent + SQLite short-term memory"`
- Do NOT push. Leave pushing to me after review.
