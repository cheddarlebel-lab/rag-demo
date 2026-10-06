"""Retrievers: dense (nomic-embed-text), lexical (BM25) and hybrid (reciprocal-rank fusion).

The eval (eval/results.json) showed BM25 alone beating dense embeddings on
section-level hits for this corpus (API names like `MAX_CONTENT_LENGTH` are exact
tokens), and RRF fusion beating both - so hybrid is the default used by the pipeline.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from . import config
from .embed import embed_query
from .index import Index, IndexValidationError

TOKEN = re.compile(r"[a-z0-9_]+")


def _tok(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def validate_k(k: Any, name: str = "k") -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"{name} must be a positive integer, got {k!r}")
    return k


def validate_mode(mode: str) -> str:
    if mode not in config.RETRIEVERS:
        raise ValueError(f"unknown retriever mode {mode!r}; expected one of {', '.join(config.RETRIEVERS)}")
    return mode


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise IndexValidationError("BM25 needs at least one document; the corpus is empty")
        self.toks = [_tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = max(sum(map(len, self.toks)) / len(self.toks), 1e-9)
        df = Counter(t for d in self.toks for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tfs = [Counter(d) for d in self.toks]

    def scores(self, q: str) -> np.ndarray:
        qt = _tok(q)
        out: np.ndarray = np.zeros(len(self.toks), dtype=np.float64)
        for i, (tf, d) in enumerate(zip(self.tfs, self.toks, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in qt if t in tf)
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    score: dict[int, float] = {}
    for lst in rankings:
        for r, i in enumerate(lst):
            score[i] = score.get(i, 0.0) + 1 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])  # stable: ties keep first-seen order


class Retriever:
    def __init__(self, index: Index | None = None):
        self.index = index or Index()
        self.bm25 = BM25([f"{c['breadcrumb']} {c['text']}" for c in self.index.chunks])

    def rankings(self, query: str, depth: int) -> dict[str, list[int]]:
        """Top-`depth` chunk indices for every retriever (used by the eval)."""
        validate_k(depth, "depth")
        dense = [int(i) for i in np.argsort(-(self.index.vecs @ embed_query(query)))[:depth]]
        lexical = [int(i) for i in np.argsort(-self.bm25.scores(query))[:depth]]
        return {"dense": dense, "bm25": lexical, "hybrid": rrf([dense, lexical])[:depth]}

    def search(self, query: str, k: int = config.TOP_K, mode: str = config.RETRIEVER,
               depth: int | None = None) -> list[dict[str, Any]]:
        validate_k(k)
        validate_mode(mode)
        depth = max(validate_k(config.CANDIDATE_DEPTH if depth is None else depth, "depth"), k)
        if mode == "dense":
            order = [int(i) for i in np.argsort(-(self.index.vecs @ embed_query(query)))[:depth]]
        elif mode == "bm25":
            order = [int(i) for i in np.argsort(-self.bm25.scores(query))[:depth]]
        else:
            order = self.rankings(query, depth)["hybrid"]
        return [{**self.index.chunks[i], "rank": r + 1} for r, i in enumerate(order[:k])]
