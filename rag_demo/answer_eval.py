"""End-to-end (retrieval + LLM) grounding eval.

Runs the full pipeline over every labelled question plus a few out-of-corpus
"trap" questions, and measures what can be checked mechanically - no LLM judge:

  answered_with_citations - answer has >=1 citation pointing at a retrieved passage
  quote_match_rate        - of the citations where the model supplied a quote, the share whose
                            quote occurs in the cited passage after whitespace/case/backtick
                            normalization
  cites_labelled_target   - share of in-corpus answers citing at least one labelled target page
  trap_abstain_rate       - share of out-of-corpus questions where the model set insufficient_context

Output errors (schema violations, truncation, refusals) are counted per row instead of
aborting the run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from typing import Any

from . import config
from .evaluate import load_labels, questions_fingerprint
from .llm import LLMOutputError, get_llm
from .log import event, get_logger
from .manifest import run_manifest
from .pipeline import answer

log = get_logger(__name__)

TRAPS = [
    "How do I configure middleware ordering in Django settings?",
    "What is the default port for a PostgreSQL server?",
    "How do I define a route in Express.js?",
    "Which Kubernetes object should I use to run a cron job?",
    "How do I enable server-side rendering in Next.js?",
]


def question_set_fingerprint(path: str, limit: int | None) -> str:
    """The default full file keeps the file fingerprint (comparable with the retrieval eval); a custom
    path or a --limit subset gets a fingerprint of exactly the questions that were asked, plus the traps."""
    if limit is None:
        return questions_fingerprint(path)
    asked = [lab["q"] for lab in load_labels(path)[:limit]] + TRAPS
    return "subset-" + hashlib.sha256(json.dumps(asked).encode()).hexdigest()[:16]


def answer_set_fingerprint(path: str = "eval/questions.jsonl", limit: int | None = None) -> str:
    """Identity of exactly what was asked and how it is scored: the selected label objects
    (questions *and* targets) plus the trap questions."""
    selected = load_labels(path)[:limit]
    return hashlib.sha256(json.dumps({"labels": selected, "traps": TRAPS}, sort_keys=True).encode()).hexdigest()[:16]


def summarize(rows: list[dict[str, Any]], llm_name: str, qfp: str, path: str, limit: int | None,
              wall_s: float, asfp: str) -> dict[str, Any]:
    """Rates use every *attempted* question as the denominator: a rejected model output (schema
    violation, truncation, refusal) counts as a failure, never silently drops out."""
    real = [r for r in rows if not r["trap"]]
    traps = [r for r in rows if r["trap"]]
    ok = [r for r in rows if "error" not in r]
    n_quotes = sum(r.get("quotes_given", 0) for r in real)

    def rate(rs: list[dict[str, Any]], pred: Any) -> float:
        return round(sum(1 for r in rs if "error" not in r and pred(r)) / max(len(rs), 1), 3)

    return {
        "manifest": {**run_manifest(llm=llm_name), "questions_fingerprint": qfp, "questions_path": path,
                     "limit": limit, "answer_set_fingerprint": asfp},
        "model": llm_name, "retriever": f"local/{config.RETRIEVER}", "top_k": config.TOP_K,
        "questions": len(real), "trap_questions": len(traps),
        "output_errors": len(rows) - len(ok),
        "denominator": "attempted questions; rows with output errors count as failures",
        "answered_with_citations": rate(real, lambda r: r["citations_ref_valid"] > 0),
        "inline_markers_valid": rate(real, lambda r: r["markers_valid"]),
        "quote_match_rate": round(sum(r.get("quotes_verified", 0) for r in real) / max(n_quotes, 1), 3),
        "quotes_given": n_quotes,
        "cites_labelled_target": rate(real, lambda r: r["cites_target"]),
        "false_abstain_rate": rate(real, lambda r: bool(r["insufficient_context"])),
        "trap_abstain_rate": rate(traps, lambda r: bool(r["insufficient_context"])),
        "median_llm_ms": statistics.median(r["llm_ms"] for r in ok) if ok else None,
        "wall_clock_s": round(wall_s),
    }


def run_answer_eval(path: str = "eval/questions.jsonl", out: str = "eval/answer_results.json",
                    limit: int | None = None) -> dict[str, Any]:
    labels = load_labels(path)[:limit]
    llm = get_llm()
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, lab in enumerate(labels + [{"q": q, "targets": [], "trap": True} for q in TRAPS]):
        base = {"q": lab["q"], "trap": lab.get("trap", False)}
        try:
            r = answer(lab["q"], llm=llm)
        except LLMOutputError as exc:
            rows.append({**base, "error": str(exc)})
            event(log, "answer_eval.output_error", logging.WARNING, n=i + 1, error=str(exc))
            continue
        targets = {config.ALLOWED_PREFIX + t["page"] for t in lab["targets"]}
        cited_pages = {c["link"].split("#")[0] for c in r["citations"] if c["link"]}
        rows.append({
            **base, "model": r["model"],
            "answer": r["answer"], "insufficient_context": r["insufficient_context"],
            "citations": len(r["citations"]),
            "citations_ref_valid": r["grounding"]["citations_ref_valid"],
            "quotes_given": r["grounding"]["quotes_given"],
            "quotes_verified": r["grounding"]["quotes_verified"],
            "markers_valid": r["grounding"]["markers_valid"],
            "cites_target": bool(targets & cited_pages),
            "llm_ms": r["timing_ms"]["llm"],
        })
        event(log, "answer_eval.row", n=i + 1, citations=rows[-1]["citations"],
              verified=rows[-1]["quotes_verified"], target=rows[-1]["cites_target"])

    summary = summarize(rows, llm.name, question_set_fingerprint(path, limit), path, limit, time.time() - t0,
                        answer_set_fingerprint(path, limit))
    (config.ROOT / out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    event(log, "answer_eval.done", **{k: v for k, v in summary.items() if k != "manifest"})
    return summary
