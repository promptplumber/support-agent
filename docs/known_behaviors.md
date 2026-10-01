# Known behaviors (revisit in Week 3 eval set)

## Q2 — billing dispute does not always escalate (src/react_agent.py)

- Question: "I was charged for a teammate who left, what now?"
- On an isolated thread, the hand-rolled agent calls `search_tickets`, then
  asks "Would you like me to proceed?" instead of calling `escalate_to_human`.
- `react_prebuilt.py` has a stronger "MUST call escalate" instruction for
  billing disputes; the hand-rolled system prompt does not.
- An earlier run that did escalate was on a shared `demo` thread, so leaked
  history likely influenced it. Not a reliable result.
- Decision: system prompt intentionally left unchanged for now. Add this case
  to the Week 3 eval set (expected tool: `escalate_to_human`) and measure.
