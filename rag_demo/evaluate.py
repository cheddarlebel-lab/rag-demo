"""Retrieval eval: hit-rate@k and MRR over a hand-labelled question set.

Each line of eval/questions.jsonl:
  {"q": "...", "targets": [{"page": "<path under the docs prefix>", "anchors": ["section-or-api-id", ...]}, ...]}

Flask's docs cover some topics in more than one place (e.g. debug mode is explained
in the quickstart, the CLI page and the config page), so a question may list several
acceptable targets. Labels were written by hand before looking at retrieval output.

Two strictness levels are reported:
  page@k    - any of the top-k chunks comes from an expected page
  section@k - any of the top-k chunks comes from an expected (page, anchor) pair

A lexical BM25 baseline over the exact same chunks is reported alongside the dense
retriever, so the embedding numbers have something honest to be compared against.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import config
from .embed import embed_query
from .index import Index
from .retrieve import BM25, rrf

def _metrics(rankings: list[list[int]], chunks, labels, ks) -> dict:
    res = {}
    for level in ("page", "section"):
        rr, hits = [], {k: 0 for k in ks}
        for order, lab in zip(rankings, labels):
            targets = {config.ALLOWED_PREFIX + t["page"]: set(t.get("anchors", [])) for t in lab["targets"]}

            def ok(i):
                c = chunks[i]
                if c["url"] not in targets:
                    return False
                return level == "page" or c["anchor"] in targets[c["url"]]

            first = next((r for r, i in enumerate(order) if ok(i)), None)
            rr.append(0 if first is None else 1 / (first + 1))
            for k in ks:
                hits[k] += first is not None and first < k
        n = len(labels)
        res[level] = {**{f"hit@{k}": round(hits[k] / n, 3) for k in ks}, "mrr": round(sum(rr) / n, 3)}
    return res


def run_eval(ks=(1, 3, 5, 10), out="eval/results.json", path="eval/questions.jsonl") -> dict:
    labels = [json.loads(l) for l in open(config.ROOT / path) if l.strip()]
    idx = Index()
    chunks = idx.chunks
    depth = max(ks)

    t0 = time.perf_counter()
    dense = []
    for lab in labels:
        s = idx.vecs @ embed_query(lab["q"])
        dense.append(list(np.argsort(-s)[:50]))
    dense_ms = (time.perf_counter() - t0) * 1000 / len(labels)

    bm = BM25([f"{c['breadcrumb']} {c['text']}" for c in chunks])
    lexical = [list(np.argsort(-bm.scores(l["q"]))[:50]) for l in labels]

    # simple hybrid: reciprocal-rank fusion of the two lists (k=60, the usual constant)
    hybrid = [rrf([d, b])[:50] for d, b in zip(dense, lexical)]

    result = {
        "questions": len(labels),
        "corpus": json.loads((config.INDEX_DIR / "meta.json").read_text()),
        "dense_nomic": _metrics(dense, chunks, labels, ks),
        "bm25_baseline": _metrics(lexical, chunks, labels, ks),
        "hybrid_rrf": _metrics(hybrid, chunks, labels, ks),
        "dense_avg_query_ms_incl_embedding": round(dense_ms, 1),
        "per_question": [
            {"q": l["q"], "targets": [t["page"] for t in l["targets"]],
             "dense_top3": [chunks[i]["link"] for i in d[:3]],
             "dense_first_relevant_page_rank": next(
                 (r + 1 for r, i in enumerate(d)
                  if chunks[i]["url"] in {config.ALLOWED_PREFIX + t["page"] for t in l["targets"]}), None)}
            for l, d in zip(labels, dense)],
    }
    (config.ROOT / out).write_text(json.dumps(result, indent=2))
    for name in ("dense_nomic", "bm25_baseline", "hybrid_rrf"):
        print(f"{name:14s} page: {result[name]['page']}")
        print(f"{'':14s} sect: {result[name]['section']}")
    print(f"n={len(labels)} questions; dense query latency {result['dense_avg_query_ms_incl_embedding']} ms")
    return result
