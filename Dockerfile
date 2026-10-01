# Cloud Run image for the support agent (FastAPI + hand-rolled ReAct graph).
# NOTE: the image is large (torch + sentence-transformers + the embedding model,
# roughly 1-2 GB). That's expected -- it buys a fast cold start, because the
# model and index are baked in instead of downloaded on the first request.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first. The default Linux wheel drags in multi-GB CUDA libraries
# Cloud Run can't use. faiss-cpu ships wheels, so no compiler is needed.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements alone first so this layer is cached until requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the knowledge base + FAISS index into the image.
RUN python scripts/build_kb.py && python -m src.retrieval

# Pre-download the embedding model so the first request doesn't fetch it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Secrets (OPENAI_API_KEY, LANGFUSE_*) are NOT in the image; Cloud Run injects them at deploy.
# Cloud Run sets PORT; default to 8080 for local `docker run`.
CMD exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}
