"""Retrievers: dense (nomic-embed-text), lexical (BM25) and hybrid (reciprocal-rank fusion).

The eval (eval/results.json) showed BM25 alone beating dense embeddings on
section-level hits for this corpus (API names like `MAX_CONTENT_LENGTH` are exact
tokens), and RRF fusion beating both - so hybrid is the default used by the pipeline.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from . import config
from .embed import embed_query
from .index import Index

TOKEN = re.compile(r"[a-z0-9_]+")


def _tok(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [_tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(t for d in self.toks for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tfs = [Counter(d) for d in self.toks]

    def scores(self, q: str) -> np.ndarray:
        qt = _tok(q)
        out = np.zeros(len(self.toks))
        for i, (tf, d) in enumerate(zip(self.tfs, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in qt if t in tf)
        return out



def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    score: Counter = Counter()
    for lst in rankings:
        for r, i in enumerate(lst):
            score[i] += 1 / (k + r + 1)
    return [i for i, _ in score.most_common()]


class Retriever:
    def __init__(self, index: Index | None = None):
        self.index = index or Index()
        self.bm25 = BM25([f"{c['breadcrumb']} {c['text']}" for c in self.index.chunks])

    def search(self, query: str, k: int = config.TOP_K, mode: str = config.RETRIEVER) -> list[dict]:
        depth = 50
        dense = list(np.argsort(-(self.index.vecs @ embed_query(query)))[:depth]) if mode != "bm25" else []
        lexical = list(np.argsort(-self.bm25.scores(query))[:depth]) if mode != "dense" else []
        order = {"dense": dense, "bm25": lexical}.get(mode) or rrf([dense, lexical])
        return [{**self.index.chunks[i], "rank": r + 1} for r, i in enumerate(order[:k])]
