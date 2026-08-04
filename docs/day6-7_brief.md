# Task brief: Week 1 Day 6–7 — Langfuse tracing

Work in the `support-agent` repo. Add Langfuse tracing to the agent so every run
produces one clean trace (each node, each LLM call, tokens, cost, latency). Match
existing code style. Do not break the working agent.

## Repo context (already built — do not rewrite)
- `src/config.py` — settings from `.env`. Already loads `.env` via `load_dotenv`
  at import, so importing config makes env vars available.
- `src/rag_graph.py` — `retrieve`, `answer`, `get_llm`, `get_retriever`,
  `_format_context`. Leave untouched.
- `src/agent_graph.py` — the CURRENT agent: `START -> retrieve -> grade -> answer|rewrite`
  self-correcting loop, `AgentState` (question, chunks, answer, attempts, retrieval_ok,
  path). `main()` runs `graph.invoke(...)` and prints answer, sources, attempts, path.
- `.env` already contains (from Day 0) `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
  `LANGFUSE_HOST`. A real OpenAI key is also present.

## Versions / correct API (IMPORTANT)
Installed **langfuse 4.14.2** (v3-line SDK). Use ONLY this import style:
- `from langfuse.langchain import CallbackHandler`   ← correct
- `from langfuse import get_client, propagate_attributes`
- Do NOT use `from langfuse.callback import CallbackHandler` — that is the old v2
  API and will fail. If you see a tutorial using it, ignore it.

The handler reads keys from env automatically:
```python
from langfuse.langchain import CallbackHandler
handler = CallbackHandler()   # no args; picks up LANGFUSE_* env vars
```
Pass it to the graph as a request callback:
```python
graph.invoke(inputs, config={"callbacks": [handler]})
```
Group + name the trace and attach metadata with the context manager:
```python
from langfuse import propagate_attributes
with propagate_attributes(trace_name="support-agent", session_id=..., tags=[...], metadata={...}):
    result = graph.invoke(inputs, config={"callbacks": [handler]})
```
Flush before the process exits (CLI is short-lived, or traces won't send):
```python
from langfuse import get_client
get_client().flush()
```
Trace id for printing a link: `handler.last_trace_id`.

## What to build

### New file: `src/tracing.py`
A small, reusable tracing helper so both the CLI now and the FastAPI app later can
use it. It must **degrade gracefully**: if Langfuse keys are missing or still the
`.env.example` placeholders, tracing is skipped and the agent runs normally.

Provide:
1. `tracing_enabled() -> bool` — True only if both LANGFUSE_PUBLIC_KEY and
   LANGFUSE_SECRET_KEY are set and are not placeholder values (they start with
   `pk-lf-` / `sk-lf-`; treat `pk-lf-...` style placeholders or empty as disabled).
2. `get_handler()` — returns a `CallbackHandler()` if enabled, else `None`.
3. `run_traced(graph, inputs, *, trace_name="support-agent", session_id=None, tags=None, metadata=None) -> (result, trace_id)`:
   - If tracing disabled: just `graph.invoke(inputs)` and return `(result, None)`.
   - If enabled: create the handler, wrap the invoke in `propagate_attributes(...)`
     with the given trace_name/session_id/tags/metadata, pass
     `config={"callbacks": [handler]}`, then `get_client().flush()`, and return
     `(result, handler.last_trace_id)`.
   - Keep it defensive: any Langfuse error must NOT crash the agent. Wrap the
     tracing parts so a failure falls back to a normal untraced invoke.

### Wire into `src/agent_graph.py` `main()`
- Replace the direct `graph.invoke(...)` with `run_traced(...)`.
- Pass useful metadata AFTER the run isn't possible (metadata is set before), so
  pass what you know up front: `metadata={"input_question": question, "model": config.ANSWER_MODEL}`
  and `tags=["week1", "agent"]`. (Per-run outputs like attempts are already captured
  as node results inside the trace; no need to duplicate.)
- After the run, if a `trace_id` came back, print a line like:
  `Trace: <trace_id>  (open it in your Langfuse dashboard)`.
  If the SDK exposes a direct URL helper, use it; otherwise printing the id is fine.
- If tracing is disabled, print a one-line hint: `Tracing disabled (set LANGFUSE_* in .env)`.

## Test it (run live)
Run all three and confirm each prints a trace id (tracing is configured locally):
1. `python -m src.agent_graph "how much does the team plan cost?"`
2. `python -m src.agent_graph "what is the airspeed velocity of an unladen swallow?"`
3. `python -m src.agent_graph "how do i stop paying"`

Then open the Langfuse dashboard and confirm:
- **One trace per run** (not scattered fragments).
- Nested spans for each node: `retrieve`, `grade`, `answer` (and for run 2, the
  repeated `retrieve/grade/rewrite` spans — the loop should be visible in the trace).
- Each LLM call shows a token count and a latency. Cost may show if the model is in
  Langfuse's price list.

## Gotchas to expect (and not panic over)
- If cost shows as 0 or blank: that's just Langfuse not having a price mapped for
  `gpt-4.1-mini`. It's not a code bug. Note it; we can add a model price in Langfuse
  later. Tokens and latency are the important part now.
- If traces don't appear: you almost certainly missed the `flush()`, or the keys
  point at the wrong region host. Verify `LANGFUSE_HOST` matches the region where
  the Langfuse project lives (EU vs US).
- Do NOT print or commit the secret keys. `.env` stays gitignored.

## README
Tick the Day 6–7 checkbox and add one line: runs are traced in Langfuse (per-node
spans, tokens, latency, cost).

## When done
- Paste the output of the three commands (showing the trace ids).
- One commit: `git add . && git commit -m "Week 1 Day 6-7: Langfuse tracing (per-node spans, tokens, latency, cost)"`
- Do NOT push. Leave pushing to me after review.
