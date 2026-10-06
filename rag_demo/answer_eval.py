"""End-to-end (retrieval + LLM) grounding eval.

Runs the full pipeline over every labelled question plus a few out-of-corpus
"trap" questions, and measures what can be checked mechanically - no LLM judge:

  answered_with_citations - answer has >=1 citation pointing at a retrieved passage
  quote_verified_rate     - of the citations where the model supplied a quote, the share whose
                            quote occurs verbatim (whitespace/case-normalized) in the cited passage
  cites_labelled_target   - share of in-corpus answers citing at least one labelled target page
  trap_abstain_rate       - share of out-of-corpus questions where the model set insufficient_context
"""
from __future__ import annotations

import json
import time

from . import config
from .pipeline import answer

TRAPS = [
    "How do I configure middleware ordering in Django settings?",
    "What is the default port for a PostgreSQL server?",
    "How do I define a route in Express.js?",
    "Which Kubernetes object should I use to run a cron job?",
    "How do I enable server-side rendering in Next.js?",
]


def run_answer_eval(path="eval/questions.jsonl", out="eval/answer_results.json", limit=None) -> dict:
    labels = [json.loads(l) for l in open(config.ROOT / path) if l.strip()][:limit]
    rows = []
    t0 = time.time()
    for i, lab in enumerate(labels + [{"q": q, "targets": [], "trap": True} for q in TRAPS]):
        r = answer(lab["q"])
        targets = {config.ALLOWED_PREFIX + t["page"] for t in lab["targets"]}
        cited_pages = {c["link"].split("#")[0] for c in r["citations"] if c["link"]}
        rows.append({
            "q": lab["q"], "trap": lab.get("trap", False),
            "answer": r["answer"], "insufficient_context": r["insufficient_context"],
            "citations": len(r["citations"]),
            "citations_ref_valid": r["grounding"]["citations_ref_valid"],
            "quotes_given": r["grounding"]["quotes_given"],
            "quotes_verified": r["grounding"]["quotes_verified"],
            "markers_valid": r["grounding"]["markers_valid"],
            "cites_target": bool(targets & cited_pages),
            "llm_ms": r["timing_ms"]["llm"],
        })
        print(f"[{i + 1}] {lab['q'][:60]:60s} cites={rows[-1]['citations']} "
              f"verified={rows[-1]['quotes_verified']} target={rows[-1]['cites_target']}", flush=True)

    real = [r for r in rows if not r["trap"]]
    traps = [r for r in rows if r["trap"]]
    n_cit = sum(r["quotes_given"] for r in real)
    summary = {
        "model": config.CHAT_MODEL, "retriever": config.RETRIEVER, "top_k": config.TOP_K,
        "questions": len(real), "trap_questions": len(traps),
        "answered_with_citations": round(sum(r["citations_ref_valid"] > 0 for r in real) / len(real), 3),
        "inline_markers_valid": round(sum(r["markers_valid"] for r in real) / len(real), 3),
        "quote_verified_rate": round(sum(r["quotes_verified"] for r in real) / max(n_cit, 1), 3),
        "cites_labelled_target": round(sum(r["cites_target"] for r in real) / len(real), 3),
        "false_abstain_rate": round(sum(bool(r["insufficient_context"]) for r in real) / len(real), 3),
        "trap_abstain_rate": round(sum(bool(r["insufficient_context"]) for r in traps) / len(traps), 3),
        "median_llm_ms": sorted(r["llm_ms"] for r in rows)[len(rows) // 2],
        "wall_clock_s": round(time.time() - t0),
    }
    (config.ROOT / out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return summary
