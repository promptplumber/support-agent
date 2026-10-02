"""
One place for settings. Everything tunable is read from .env so you can change
models or top-k without editing code — same pattern as your work pipelines.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root into environment variables (once, at import).
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- Paths ---
KB_DIR = ROOT / "knowledge_base"          # the 34 source docs
INDEX_DIR = KB_DIR / "index"              # built FAISS index (gitignored)

# --- Models (override in .env) ---
# The answer model. gpt-4.1-mini is the default; swap in .env to test others.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")

# Ceiling on tokens the model may WRITE per call. Answers are short, so this only
# stops a manipulated or runaway generation from burning money; one-word calls
# (the guardrail's yes/no) stay short on their own.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "500"))

# The embedding model used for retrieval. bge-small is small, fast, and good.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Retrieval settings ---
TOP_K = int(os.getenv("TOP_K", "5"))      # how many chunks to pull per question
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "800"))   # target chunk size
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # overlap between chunks

# --- Self-correcting retrieval loop (Day 5) ---
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))   # original attempt + up to 2 rewrites

# grade_retrieval's cheap guard: a top score at/above HIGH is clearly good, a
# top score below LOW is clearly bad -- skip the LLM call in both cases and
# only pay for grading in the ambiguous middle band. Measured gap on this
# corpus: answerable questions score ~0.65-0.70, off-topic ones ~0.45 or below.
GRADE_HIGH_SCORE = float(os.getenv("GRADE_HIGH_SCORE", "0.65"))
GRADE_LOW_SCORE = float(os.getenv("GRADE_LOW_SCORE", "0.45"))


def require_openai_key() -> str:
    """Fail loudly and early if the OpenAI key is missing, with a clear message."""
    key = os.getenv("OPENAI_API_KEY")
    if not key or key.startswith("sk-...") or key == "":
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key."
        )
    return key
