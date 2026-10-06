"""Retrieval eval: hit-rate@k and MRR@depth over a hand-labelled question set.

Each line of eval/questions.jsonl:
  {"q": "...", "targets": [{"page": "<path under the docs prefix>", "anchors": ["section-or-api-id", ...]}, ...]}

Flask's docs cover some topics in more than one place (e.g. debug mode is explained
in the quickstart, the CLI page and the config page), so a question may list several
acceptable targets. Labels were written by hand before looking at retrieval output.

Two strictness levels are reported:
  page@k    - any of the top-k results comes from an expected page
  section@k - any of the top-k results comes from an expected (page, anchor) pair

MRR is computed over the top `depth` results (default 50) and labelled mrr@<depth>.
The same metric code scores the AnythingLLM workspace retrieval (anythingllm.py).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import config
from .index import Index, fingerprint
from .log import event, get_logger
from .retrieve import Retriever, validate_k

log = get_logger(__name__)

Hit = tuple[str, str]  # (page url, anchor)


def load_labels(path: str = "eval/questions.jsonl") -> list[dict[str, Any]]:
    with open(config.ROOT / path, encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    if not labels:
        raise ValueError(f"{path} contains no questions")
    return labels


def questions_fingerprint(path: str = "eval/questions.jsonl") -> str:
    return hashlib.sha256((config.ROOT / path).read_bytes()).hexdigest()[:16]


def first_relevant(hits: list[Hit], label: dict[str, Any], level: str) -> int | None:
    """1-based rank of the first relevant hit, or None."""
    targets = {config.ALLOWED_PREFIX + t["page"]: set(t.get("anchors", [])) for t in label["targets"]}
    for r, (url, anchor) in enumerate(hits):
        if url in targets and (level == "page" or anchor in targets[url]):
            return r + 1
    return None


def metrics(rankings: list[list[Hit]], labels: list[dict[str, Any]], ks: list[int], depth: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    n = len(labels)
    for level in ("page", "section"):
        firsts = [first_relevant(h[:depth], lab, level) for h, lab in zip(rankings, labels, strict=True)]
        res[level] = {**{f"hit@{k}": round(sum(f is not None and f <= k for f in firsts) / n, 3) for k in ks},
                      f"mrr@{depth}": round(sum(0 if f is None else 1 / f for f in firsts) / n, 3)}
    return res


def run_eval(ks: list[int] | tuple[int, ...] = (1, 3, 5, 10), out: str = "eval/results.json",
             path: str = "eval/questions.jsonl", depth: int | None = None) -> dict[str, Any]:
    ks = [validate_k(k, "ks entry") for k in ks]
    if not ks:
        raise ValueError("ks must contain at least one positive integer")
    depth = max(validate_k(config.CANDIDATE_DEPTH if depth is None else depth, "depth"), max(ks))
    labels = load_labels(path)
    idx = Index()
    retriever = Retriever(idx)
    chunks = idx.chunks

    t0 = time.perf_counter()
    ranked = [retriever.rankings(lab["q"], depth) for lab in labels]
    avg_ms = (time.perf_counter() - t0) * 1000 / len(labels)

    def hits(name: str) -> list[list[Hit]]:
        return [[(chunks[i]["url"], chunks[i]["anchor"]) for i in r[name]] for r in ranked]

    names = {"dense": "dense_nomic", "bm25": "bm25_baseline", "hybrid": "hybrid_rrf"}
    per_question = []
    for lab, r in zip(labels, ranked, strict=True):
        row: dict[str, Any] = {"q": lab["q"], "targets": [t["page"] for t in lab["targets"]]}
        for name in names:
            h = [(chunks[i]["url"], chunks[i]["anchor"]) for i in r[name]]
            row[name] = {"top5": [chunks[i]["link"] for i in r[name][:5]],
                         "first_page_rank": first_relevant(h, lab, "page"),
                         "first_section_rank": first_relevant(h, lab, "section")}
        per_question.append(row)

    result: dict[str, Any] = {
        "questions": len(labels),
        "manifest": {"corpus_fingerprint": fingerprint(config.INDEX_DIR / "chunks.jsonl"),
                     "questions_fingerprint": questions_fingerprint(path),
                     "embed_model": idx.meta["embed_model"], "candidate_depth": depth, "ks": ks,
                     "generated": time.strftime("%Y-%m-%d")},
        "corpus": idx.meta,
        **{names[n]: metrics(hits(n), labels, ks, depth) for n in names},
        "avg_query_ms_all_three_retrievers_incl_embedding": round(avg_ms, 1),
        "per_question": per_question,
    }
    Path(config.ROOT / out).write_text(json.dumps(result, indent=2))
    for n in names.values():
        event(log, "eval.retrieval", retriever=n, page=result[n]["page"], section=result[n]["section"])
    return result
