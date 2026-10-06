"""Retrieval -> analysis -> validated, cited JSON.

The LLM sees numbered passages and must answer only from them, citing by number
with a verbatim supporting quote. We then *check* every citation in code:
the ref must point at a retrieved passage, and the quote must actually occur in
that passage. Unverifiable citations are kept but flagged, never silently trusted.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache

from . import config
from .retrieve import Retriever
from .llm import LLM, get_llm

SYSTEM = """You are a documentation analyst. Answer the user's question using ONLY the numbered passages provided.
Rules:
- Write a complete, practical answer of 2-5 sentences; include the relevant config key, function or code snippet when the passages give one.
- Every factual claim in "answer" must carry a citation marker like [1] or [2][3] that refers to a passage number.
- "citations" lists each passage you relied on, with a short quote copied VERBATIM from that passage (under 200 characters).
- If the passages do not contain the answer, set "insufficient_context" to true, say so plainly in "answer", and do not guess.
- "confidence" is "high" only when a passage states the answer directly.
Citation style example (unrelated topic): "The cache entry expires after ten minutes [2]; change it with the CACHE_TTL setting [2][4]." """

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {
            "type": "object",
            "properties": {"ref": {"type": "integer"}, "quote": {"type": "string"}},
            "required": ["ref", "quote"], "additionalProperties": False}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "insufficient_context": {"type": "boolean"},
    },
    "required": ["answer", "key_points", "citations", "confidence", "insufficient_context"],
    "additionalProperties": False,
}


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("`", "")).strip().lower()


def _passage(c: dict) -> dict:
    return {k: c[k] for k in ("rank", "link", "breadcrumb", "kind", "text")}


def retrieve_only(question: str, k: int | None = None) -> dict:
    t0 = time.perf_counter()
    hits = _retriever().search(question, k=k or config.TOP_K)
    return {"question": question, "retriever": config.RETRIEVER, "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1),
            "passages": [_passage(h) for h in hits]}


def answer(question: str, k: int | None = None, llm: LLM | None = None) -> dict:
    llm = llm or get_llm()
    ret = retrieve_only(question, k)
    passages = ret["passages"]
    context = "\n\n".join(
        f"[{p['rank']}] ({p['breadcrumb']})\n{p['text']}" for p in passages)
    user = f"Passages:\n\n{context}\n\nQuestion: {question}"

    t0 = time.perf_counter()
    raw = llm.complete_json(SYSTEM, user, SCHEMA)
    llm_ms = round((time.perf_counter() - t0) * 1000)

    by_rank = {p["rank"]: p for p in passages}
    citations = []
    for c in raw.get("citations", []):
        p = by_rank.get(c.get("ref"))
        quote = (c.get("quote") or "").strip()
        citations.append({
            "ref": c.get("ref"),
            "link": p["link"] if p else None,
            "breadcrumb": p["breadcrumb"] if p else None,
            "quote": quote,
            "ref_valid": p is not None,
            "quote_verified": bool(p and quote and _norm(quote) in _norm(p["text"])),
        })
    markers = sorted({int(m) for m in re.findall(r"\[(\d+)\]", raw.get("answer", ""))})
    # If the model used inline [n] markers but left the citations array empty or partial,
    # add the missing refs as citations without a quote (flagged, never counted as verified).
    for m in markers:
        if m in by_rank and all(c["ref"] != m for c in citations):
            p = by_rank[m]
            citations.append({"ref": m, "link": p["link"], "breadcrumb": p["breadcrumb"], "quote": "",
                              "ref_valid": True, "quote_verified": False, "from_inline_marker": True})
    return {
        "question": question,
        "answer": raw.get("answer", ""),
        "key_points": raw.get("key_points", []),
        "citations": citations,
        "confidence": raw.get("confidence"),
        "insufficient_context": raw.get("insufficient_context"),
        "grounding": {
            "citations_total": len(citations),
            "citations_ref_valid": sum(c["ref_valid"] for c in citations),
            "quotes_given": sum(bool(c["quote"]) for c in citations),
            "quotes_verified": sum(c["quote_verified"] for c in citations),
            "answer_markers": markers,
            "answer_has_markers": bool(markers),
            "markers_valid": bool(markers) and all(m in by_rank for m in markers),
        },
        "model": llm.name,
        "retriever": ret["retriever"],
        "timing_ms": {"retrieval": ret["retrieval_ms"], "llm": llm_ms},
        "passages": passages,
    }
