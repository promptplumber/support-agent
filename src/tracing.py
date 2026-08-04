"""
Week 1, Day 6-7 — Langfuse tracing.

A small, reusable wrapper around Langfuse so the CLI (now) and the FastAPI
app (Week 2) can both trace a graph run the same way. It degrades gracefully:
if the Langfuse keys in .env are missing or still the .env.example
placeholders, tracing is skipped and the agent just runs untraced.

Uses the langfuse 4.x (v3 SDK) API:
  from langfuse.langchain import CallbackHandler
  from langfuse import get_client, propagate_attributes
"""

from __future__ import annotations

import os


def _is_placeholder(value: str) -> bool:
    # .env.example ships "pk-lf-..." / "sk-lf-..." -- a real key never has
    # literal dots in it, so this is enough to tell a placeholder from a key.
    return not value or "..." in value


def tracing_enabled() -> bool:
    """True only if both Langfuse keys are set to real (non-placeholder) values."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    if _is_placeholder(pk) or _is_placeholder(sk):
        return False
    return pk.startswith("pk-lf-") and sk.startswith("sk-lf-")


def get_handler():
    """A CallbackHandler if tracing is enabled, else None."""
    if not tracing_enabled():
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def run_traced(graph, inputs, *, trace_name="support-agent", session_id=None,
                tags=None, metadata=None):
    """Invoke `graph` and, if Langfuse is configured, wrap the run in one trace.

    Returns (result, trace_id). trace_id is None when tracing is disabled or
    anything about tracing itself goes wrong -- a Langfuse problem should
    never take down the agent, so any failure here just falls back to a
    plain untraced invoke.
    """
    if not tracing_enabled():
        return graph.invoke(inputs), None

    try:
        from langfuse import get_client, propagate_attributes

        handler = get_handler()
        with propagate_attributes(
            trace_name=trace_name,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
        ):
            result = graph.invoke(inputs, config={"callbacks": [handler]})
        get_client().flush()  # CLI is short-lived -- flush now or traces never send
        return result, handler.last_trace_id
    except Exception as e:
        print(f"Tracing failed ({e}); falling back to an untraced run.")
        return graph.invoke(inputs), None
