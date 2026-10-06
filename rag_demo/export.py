"""Precompute everything the static results page shows.

The public site never calls a model: this script runs the pipeline locally and
writes site/data.json, which the page renders client-side. It refuses to combine
artifacts produced from a different corpus, embedding model, retriever or LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config
from .llm import get_llm
from .log import event, get_logger
from .manifest import IncompatibleArtifacts, require_compatible, run_manifest
from .pipeline import answer

log = get_logger(__name__)

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


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def check_anythingllm_artifacts(ingest: dict[str, Any] | None, results: dict[str, Any] | None,
                                example: dict[str, Any] | None, current: dict[str, Any],
                                rerun: dict[str, Any] | None = None) -> None:
    """The workspace must hold exactly this corpus and nothing else (verified content fingerprint, zero
    foreign documents), and every AnythingLLM artifact must come from that same workspace content,
    workspace settings, instance settings and version."""
    if not ingest:
        raise IncompatibleArtifacts("AnythingLLM artifacts present but eval/anythingllm_ingest.json is missing")
    keys = ("corpus_fingerprint", "workspace", "anythingllm_version", "workspace_content_fingerprint",
            "foreign_documents", "workspace_settings", "instance_settings")
    for name, rec in (("eval/anythingllm_ingest.json", ingest), ("eval/anythingllm_ingest_rerun.json", rerun)):
        if rec is None:
            continue
        if rec.get("status") != "complete":
            raise IncompatibleArtifacts(f"{name} has status {rec.get('status')!r}; re-run ingest")
        require_compatible(name, rec, current, ("corpus_fingerprint",))
        if rec.get("workspace_content_fingerprint") != rec.get("expected_content_fingerprint"):
            raise IncompatibleArtifacts(f"{name}: the workspace content does not match the corpus that was ingested")
        if rec.get("foreign_documents") != 0:
            raise IncompatibleArtifacts(f"{name}: the workspace also embeds documents this corpus does not own")
    ref = {k: ingest.get(k) for k in keys}
    ref["corpus_fingerprint"] = current["corpus_fingerprint"]
    ref["questions_fingerprint"] = current["questions_fingerprint"]
    ref["analysis_llm"] = current["llm"]
    if rerun:
        require_compatible("eval/anythingllm_ingest_rerun.json", rerun, ref, keys)
    if results:
        require_compatible("eval/anythingllm_results.json", results.get("manifest", {}), ref,
                           (*keys, "questions_fingerprint"))
    if example:
        require_compatible("eval/anythingllm_example.json", example.get("manifest", {}), ref, (*keys, "analysis_llm"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON, creating the parent directory (e.g. site/ inside the Docker image)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def export_site(out: str = "site/data.json") -> dict[str, Any]:
    if not config.CORPUS_NAME or not config.CORPUS_LICENSE:
        raise IncompatibleArtifacts("non-default corpus: set RAG_CORPUS_NAME and RAG_CORPUS_LICENSE explicitly")
    llm = get_llm()
    current = run_manifest(llm=llm.name)

    ret = _read(config.ROOT / "eval/results.json")
    if ret is None:
        raise IncompatibleArtifacts("eval/results.json missing; run `python -m rag_demo eval` first")
    require_compatible("eval/results.json", ret.get("manifest", {}), current,
                       ("corpus_fingerprint", "embed_model", "candidate_depth", "questions_fingerprint"))
    ans_doc = _read(config.ROOT / "eval/answer_results.json")
    ans = ans_doc["summary"] if ans_doc else None
    if ans:
        from .answer_eval import answer_set_fingerprint
        current_answers = {**current, "answer_set_fingerprint": answer_set_fingerprint()}
        require_compatible("eval/answer_results.json", ans.get("manifest", {}), current_answers,
                           ("corpus_fingerprint", "embed_model", "retriever", "top_k", "candidate_depth",
                            "questions_fingerprint", "answer_set_fingerprint", "llm"))
    alm = _read(config.ROOT / "eval/anythingllm_results.json")
    alm_example = _read(config.ROOT / "eval/anythingllm_example.json")
    alm_ingest = _read(config.ROOT / "eval/anythingllm_ingest.json")
    alm_rerun = _read(config.ROOT / "eval/anythingllm_ingest_rerun.json")
    if alm or alm_example:
        check_anythingllm_artifacts(alm_ingest, alm, alm_example, current, alm_rerun)

    examples = []
    for q in EXAMPLES:
        event(log, "export.example", question=q)
        examples.append(answer(q, llm=llm))
    models = {e["model"] for e in examples}
    crawl = json.loads((config.RAW_DIR / "manifest.json").read_text())
    data = {
        "generated": current["generated"],
        "manifest": current,
        "corpus": {
            "name": config.CORPUS_NAME,
            "source": config.ALLOWED_PREFIX,
            "license": config.CORPUS_LICENSE,
            "notice": "THIRD_PARTY_NOTICES.md",
            "crawled_at": crawl.get("crawled_at"),
            "crawl_delay_s": crawl["delay_s"],
            "skipped_by_robots": crawl["skipped_by_robots"],
            **ret["corpus"],
        },
        "stack": {"embeddings": f"ollama/{config.EMBED_MODEL}", "llm": ", ".join(sorted(models)),
                  "retriever": current["retriever"], "top_k": config.TOP_K},
        "retrieval_eval": {k: ret[k] for k in ("questions", "dense_nomic", "bm25_baseline", "hybrid_rrf",
                                                "avg_query_ms_all_three_retrievers_incl_embedding")},
        "candidate_depth": ret["manifest"]["candidate_depth"],
        "answer_eval": ans,
        "anythingllm_eval": {k: v for k, v in alm.items() if k != "per_question"} if alm else None,
        "anythingllm_ingest": {k: v for k, v in alm_ingest.items() if k != "failures"} if alm_ingest else None,
        "anythingllm_ingest_rerun_s": alm_rerun.get("wall_clock_s") if alm_rerun else None,
        "anythingllm_example": alm_example,
        "examples": examples,
    }
    write_json(config.ROOT / out, data)
    event(log, "export.written", path=out, examples=len(examples))
    return data
