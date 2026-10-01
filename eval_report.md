# Evaluation results

- Run date: 2026-10-02
- Agent model: `gpt-4.1-mini` (judge uses the same model)
- Questions: 30 (10 docs, 10 tickets, 5 escalate, 5 offtopic)

| Metric | Result |
|---|---|
| Trajectory accuracy (overall) | 27/30 (90%) |
| &nbsp;&nbsp;docs | 10/10 (100%) |
| &nbsp;&nbsp;tickets | 10/10 (100%) |
| &nbsp;&nbsp;escalate | 2/5 (40%) |
| &nbsp;&nbsp;offtopic | 5/5 (100%) |
| Escalation correct | 2/5 |
| Off-topic refusal correct | 5/5 |
| Tool rounds (avg / max) | 0.83 / 1 |
| Answer quality (docs+tickets, n=20) | 18 correct, 2 partial, 0 wrong |
| Avg cost / question (estimated) | $0.00052 |
| Latency p50 / p95 (warm) | 2.8s / 3.6s |

Notes
- Cost is estimated from token counts at gpt-4.1-mini list prices ($0.4/1M in, $1.6/1M out). It includes the guardrail call; it excludes the judge.
- Each question runs on a fresh thread (no memory carry-over); one warmup run precedes timing.

## Misses

**Escalation gap (e1, e3, e4):** on these the agent answered from the docs instead of calling `escalate_to_human`. This is the soft-escalation behavior logged in `docs/known_behaviors.md` (Q2), now measured. The system prompt was not tuned for this run.

- **d3** (docs): How many members can I have on the free plan? —  quality=partial — tools: ['search_docs']
- **t3** (tickets): The delete-project option is greyed out for me. Why? —  quality=partial — tools: ['search_docs']
- **e1** (escalate): I want a refund but it's been 20 days since I paid. — trajectory — tools: ['search_docs']
- **e3** (escalate): One of our API keys was committed to a public repo. Help. — trajectory — tools: ['search_docs']
- **e4** (escalate): I lost my phone and can't get past the 2FA code. I never saved backup codes. — trajectory — tools: ['search_docs']
