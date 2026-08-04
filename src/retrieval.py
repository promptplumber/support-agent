"""
Retrieval: turn the 34 docs into a searchable FAISS index, and look things up.

Flow:
  docs -> chunks -> embeddings -> FAISS index (saved to disk)
  question -> embedding -> nearest chunks

Build the index once:   python -m src.retrieval
Then the graph uses Retriever().search(question) at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Slide a window over the text so long docs become several overlapping
    pieces. Overlap keeps a sentence that straddles a boundary findable in at
    least one chunk. Short docs (like ours) come back as a single chunk."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap  # step forward, leaving `overlap` behind
    return chunks


def load_chunks() -> list[dict]:
    """Read every .md in knowledge_base and cut it into chunks with metadata."""
    out: list[dict] = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(chunk_text(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP)):
            out.append({"source": path.name, "chunk": i, "text": piece})
    return out


# ---------------------------------------------------------------------------
# Embedder (real model lazy-loaded so this file imports without torch present)
# ---------------------------------------------------------------------------
# bge models retrieve best when the QUERY (not the passages) is prefixed with a
# short instruction. We prefix only queries, never the stored passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        # Imported here, not at top, so the module loads even before
        # sentence-transformers is installed. Real load happens on first use.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([BGE_QUERY_PREFIX + text])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True)
        return vecs.astype("float32")


# ---------------------------------------------------------------------------
# Build + save the index
# ---------------------------------------------------------------------------
def build_index(embedder=None) -> None:
    """Embed all chunks and save a FAISS index + the chunk metadata to disk."""
    embedder = embedder or SentenceTransformerEmbedder()
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    vectors = embedder.embed_passages(texts)
    faiss.normalize_L2(vectors)  # normalize so inner-product == cosine similarity

    # IndexFlatIP = exact nearest-neighbour search by inner product. Perfect for
    # a small corpus like ours; you'd swap to an approximate index at big scale.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len(list(config.KB_DIR.glob('*.md')))} docs.")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class Retriever:
    """Loads the saved index once, then answers search() calls."""

    def __init__(self, embedder=None):
        index_path = config.INDEX_DIR / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(
                "No index found. Build it first with: python -m src.retrieval"
            )
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text("utf-8"))
        self.embedder = embedder or SentenceTransformerEmbedder()

    def search(self, question: str, k: int = config.TOP_K) -> list[dict]:
        q = self.embedder.embed_query(question)
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)   # returns top-k per query row
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:          # faiss returns -1 to pad if fewer than k exist
                continue
            hit = dict(self.chunks[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results


if __name__ == "__main__":
    build_index()
