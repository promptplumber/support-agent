"""
Week 2 Part 2c — serve the agent over HTTP.

    POST /chat     {"session_id": "...", "message": "..."}  -> the agent's answer
    GET  /         a minimal HTML chat page
    GET  /healthz  health check (Cloud Run)

Run:  python -m src.api      (or: uvicorn src.api:app --port 8080)
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from .react_agent import build_graph
from .tracing import run_traced

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

# Per-IP rate limit: stops a stranger from draining the OpenAI key.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
WINDOW_S = 60
_hits: dict[str, deque] = defaultdict(deque)   # ip -> recent request timestamps


# ---------------------------------------------------------------------------
# Startup: build everything ONCE so the first request is warm
# ---------------------------------------------------------------------------
# The CLI wraps SqliteSaver in a `with` block that closes when the run ends. A
# server lives for hours, so we open the connection once and keep it.
# check_same_thread=False because FastAPI runs sync endpoints in a thread pool.
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/memory.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    app.state.graph = build_graph(checkpointer=saver)
    # Warm the embedding model + FAISS index now rather than on the first user's request.
    from .tools import _get_doc_retriever
    _get_doc_retriever().search("warmup", 1)
    yield
    conn.close()


app = FastAPI(title="support-agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    # Cap the length too: an enormous message is just a way to burn tokens.
    message: str = Field(min_length=1, max_length=2000)


def _client_ip(request: Request) -> str:
    # On Cloud Run the real client IP is the LAST entry the platform appended to
    # X-Forwarded-For (earlier entries can be forged by the caller).
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > WINDOW_S:   # forget requests older than the window
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, slow down.")
    q.append(now)


@app.post("/chat")
def chat(body: ChatRequest, request: Request) -> dict:
    check_rate_limit(_client_ip(request))
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": body.session_id}}

    # The checkpointer returns the thread's FULL history, so remember how many
    # messages existed before this turn and report only the new ones.
    prior = graph.get_state(config)
    prior_count = len(prior.values.get("messages", [])) if prior.values else 0

    result, trace_id = run_traced(
        graph,
        {"messages": [{"role": "user", "content": body.message}], "steps": 0},
        trace_name="react-agent",
        session_id=body.session_id,
        tags=["api"],
        config=config,
    )

    new_messages = result["messages"][prior_count:]
    tools_used = [
        c["name"]
        for m in new_messages
        if isinstance(m, AIMessage)
        for c in (m.tool_calls or [])
    ]
    return {
        "answer": result["messages"][-1].content,
        "tools_used": tools_used,
        "tool_rounds": result.get("steps", 0),
        "trace_id": trace_id,
    }


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
