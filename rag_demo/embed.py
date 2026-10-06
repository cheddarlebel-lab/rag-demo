"""Local embeddings via ollama (nomic-embed-text). No API keys, no cost.

nomic-embed-text is trained with task prefixes; documents and queries must use
different ones or retrieval quality drops noticeably.
"""
from __future__ import annotations

import numpy as np
import requests

from . import config

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _embed(texts: list[str]) -> np.ndarray:
    resp = requests.post(f"{config.OLLAMA_HOST}/api/embed",
                         json={"model": config.EMBED_MODEL, "input": texts}, timeout=300)
    resp.raise_for_status()
    vecs = np.asarray(resp.json()["embeddings"], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # unit-normalize -> dot == cosine


def embed_documents(texts: list[str], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        out.append(_embed([DOC_PREFIX + t for t in texts[i:i + batch]]))
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(out)


def embed_query(text: str) -> np.ndarray:
    return _embed([QUERY_PREFIX + text])[0]
