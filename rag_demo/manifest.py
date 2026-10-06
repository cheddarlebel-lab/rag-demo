"""Run manifests: tie every generated artifact to the corpus, models and settings that made it.

Exports refuse to combine artifacts whose manifests disagree (e.g. examples from a
fresh index next to an eval summary measured on an older one).
"""
from __future__ import annotations

import time
from typing import Any

from . import __version__, config
from .evaluate import questions_fingerprint
from .index import fingerprint


class IncompatibleArtifacts(RuntimeError):
    pass


def run_manifest(llm: str | None = None, retriever: str | None = None) -> dict[str, Any]:
    return {
        "rag_demo_version": __version__,
        "corpus_fingerprint": fingerprint(config.INDEX_DIR / "chunks.jsonl"),
        "embed_model": config.EMBED_MODEL,
        "retriever": retriever or f"local/{config.RETRIEVER}",
        "top_k": config.TOP_K,
        "candidate_depth": config.CANDIDATE_DEPTH,
        "questions_fingerprint": questions_fingerprint(),
        "llm": llm,
        "generated": time.strftime("%Y-%m-%d"),
    }


def require_compatible(name: str, artifact: dict[str, Any], current: dict[str, Any],
                       keys: tuple[str, ...]) -> None:
    diffs = {k: (artifact.get(k), current.get(k)) for k in keys if artifact.get(k) != current.get(k)}
    if diffs:
        detail = ", ".join(f"{k}: artifact={a!r} current={c!r}" for k, (a, c) in diffs.items())
        raise IncompatibleArtifacts(f"{name} was produced with different settings ({detail}); regenerate it")
