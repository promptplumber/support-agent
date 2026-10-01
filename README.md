# support-agent

A live AI support agent that answers product questions from a documentation set.
Built with LangGraph. It retrieves from docs, self-corrects when retrieval is
weak, uses tools, escalates when stuck, and is fully traced and evaluated.

**Stack:** LangGraph · ReAct tool use · SQLite memory · Langfuse tracing ·
agent evals · FastAPI · GCP Cloud Run.

> Status: **in progress** — see the checklist below.

---

## What it does

_(filled in as features land)_

## Architecture

_(graph diagram goes here — LangGraph renders its own graph)_

## Evaluation

_(eval table goes here in Week 3: tool-choice accuracy, steps, escalation,
cost/question, latency)_

## Live demo

_(Cloud Run URL goes here in Week 3)_

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_kb.py           # generate the knowledge base
cp .env.example .env                 # then fill in your keys
python -m src.hello_graph            # Week 1 Day 1 hello-world graph

python -m src.retrieval              # build the FAISS index (once)
python -m src.rag_graph "how much does the team plan cost?"   # ask a question
```

---

## Progress

- [x] **Day 0** — repo, deps, fake knowledge base (34 docs), env + config
- [x] **Week 1 Day 1** — first LangGraph graph (state, nodes, edges)
- [x] **Week 1 Day 3–4** — RAG node (FAISS retrieval + grounded answer)
- [x] **Week 1 Day 5** — self-correcting retrieval (conditional edge): the
      agent grades its own retrieval and rewrites the query up to twice
      before answering, instead of always answering after one pass.
- [x] **Week 1 Day 6–7** — Langfuse tracing: every run is traced (per-node
      spans, tokens, latency, cost).
- [x] **Week 2 Part 1** — tools (docs/tickets/escalate) + prebuilt ReAct
      agent + SQLite short-term memory: tool choice and per-thread memory
      are both demonstrated live.
- [x] Week 2 Part 2a — hand-rolled ReAct loop (agent/tools/should_continue,
      step-cap escalation)
- [x] Week 2 Part 2b — guardrails (low-confidence escalation, off-topic refusal)
- [x] Week 2 Part 2c — FastAPI service + minimal chat UI
- [ ] Week 3 — evals, Cloud Run deploy, publish
