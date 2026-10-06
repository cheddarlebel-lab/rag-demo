"""Local embeddings via ollama (nomic-embed-text). No API keys, no cost.

nomic-embed-text is trained with task prefixes; documents and queries must use
different ones or retrieval quality drops noticeably.
"""
from __future__ import annotations

import numpy as np
import requests

from . import config
from .log import event, get_logger

log = get_logger(__name__)

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingError(RuntimeError):
    pass


def _embed(texts: list[str]) -> np.ndarray:
    resp = requests.post(f"{config.OLLAMA_HOST}/api/embed",
                         json={"model": config.EMBED_MODEL, "input": texts}, timeout=300)
    resp.raise_for_status()
    vecs = np.asarray(resp.json()["embeddings"], dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] != len(texts):
        raise EmbeddingError(f"expected {len(texts)} embeddings, got shape {vecs.shape}")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if not np.all(np.isfinite(vecs)) or np.any(norms == 0):
        raise EmbeddingError("embedding model returned a zero or non-finite vector")
    out: np.ndarray = vecs / norms  # unit-normalize -> dot == cosine
    return out


def embed_documents(texts: list[str], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        out.append(_embed([DOC_PREFIX + t for t in texts[i:i + batch]]))
        event(log, "embed.progress", done=min(i + batch, len(texts)), total=len(texts))
    return np.asarray(np.vstack(out))


def embed_query(text: str) -> np.ndarray:
    q: np.ndarray = _embed([QUERY_PREFIX + text])[0]
    return q
