"""Precompute everything the static results page shows.

The public site never calls a model: this script runs the pipeline locally and
writes site/data.json, which the page renders client-side.
"""
from __future__ import annotations

import datetime as dt
import json

from . import config
from .pipeline import answer

EXAMPLES = [
    "How do I limit the maximum size of an upload so large requests are rejected?",
    "My app is behind nginx and url_for generates http instead of https. How do I fix it?",
    "How do I return API errors as JSON instead of HTML?",
    "Where should I store a database connection so it is reused during one request?",
    "Does Flask include CSRF protection out of the box?",
    "How do I load configuration values from environment variables with a FLASK_ prefix?",
    "How do I run Celery background tasks with Flask?",
    "How do I configure middleware ordering in Django settings?",  # out-of-corpus trap
]


def export_site(out: str = "site/data.json") -> None:
    examples = []
    for q in EXAMPLES:
        print("->", q, flush=True)
        examples.append(answer(q))
    ret = json.loads((config.ROOT / "eval/results.json").read_text())
    ans_path = config.ROOT / "eval/answer_results.json"
    ans = json.loads(ans_path.read_text())["summary"] if ans_path.exists() else None
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text())
    data = {
        "generated": dt.date.today().isoformat(),
        "corpus": {
            "name": "Flask documentation (stable)",
            "source": config.ALLOWED_PREFIX,
            "license": "BSD-3-Clause, Copyright 2010 Pallets",
            "crawl_delay_s": manifest["delay_s"],
            "skipped_by_robots": manifest["skipped_by_robots"],
            **ret["corpus"],
        },
        "stack": {"embeddings": f"ollama/{config.EMBED_MODEL}", "llm": f"ollama/{config.CHAT_MODEL}",
                  "retriever": config.RETRIEVER, "top_k": config.TOP_K},
        "retrieval_eval": {k: ret[k] for k in ("questions", "dense_nomic", "bm25_baseline", "hybrid_rrf",
                                                "dense_avg_query_ms_incl_embedding")},
        "answer_eval": ans,
        "examples": examples,
    }
    (config.ROOT / out).write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
