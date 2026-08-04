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
```

---

## Progress

- [x] **Day 0** — repo, deps, fake knowledge base (34 docs), env + config
- [x] **Week 1 Day 1** — first LangGraph graph (state, nodes, edges)
- [ ] Week 1 Day 3–4 — RAG node
- [ ] Week 1 Day 5 — self-correcting retrieval (conditional edge)
- [ ] Week 1 Day 6–7 — Langfuse tracing
- [ ] Week 2 — tools, ReAct loop, memory, guardrails, FastAPI
- [ ] Week 3 — evals, Cloud Run deploy, publish
