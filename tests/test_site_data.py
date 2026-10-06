"""Smoke test: the committed site/data.json has every field site/index.html renders,
and the deployed license notice matches the repository's."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "site" / "data.json"


@pytest.fixture(scope="module")
def data() -> dict:  # type: ignore[type-arg]
    return json.loads(DATA.read_text())


def test_retrieval_fields(data: dict) -> None:  # type: ignore[type-arg]
    mk = f"mrr@{data['candidate_depth']}"
    for name in ("dense_nomic", "bm25_baseline", "hybrid_rrf"):
        for level in ("page", "section"):
            m = data["retrieval_eval"][name][level]
            assert {"hit@1", "hit@3", "hit@5", "hit@10", mk} <= set(m)
    assert isinstance(data["retrieval_eval"]["avg_query_ms_all_three_retrievers_incl_embedding"], float)
    assert {"chunks", "pages"} <= set(data["corpus"])


def test_answer_eval_fields(data: dict) -> None:  # type: ignore[type-arg]
    ae = data["answer_eval"]
    for k in ("quote_match_rate", "quotes_given", "output_errors", "answered_with_citations", "inline_markers_valid",
              "cites_labelled_target", "false_abstain_rate", "trap_abstain_rate", "median_llm_ms", "model",
              "questions", "trap_questions", "manifest"):
        assert k in ae, k


def test_anythingllm_fields(data: dict) -> None:  # type: ignore[type-arg]
    al = data["anythingllm_eval"]
    dk = f"mrr@{al['manifest']['depth']}"
    assert {"anythingllm_version", "generated", "route", "depth"} <= set(al["manifest"])
    for m in [al["anythingllm"], *al["local_same_depth"].values()]:
        assert dk in m["page"] and dk in m["section"]
    ingest_keys = {"embedded_in_workspace", "vector_count", "upload_failures", "wall_clock_s"}
    assert ingest_keys <= set(data["anythingllm_ingest"])
    ex = data["anythingllm_example"]
    assert ex["retriever"].startswith("anythingllm/") and "anythingllm_chat" in ex["timing_ms"]
    assert all(p["provenance"]["retriever"] == "anythingllm" for p in ex["passages"])


def test_examples_render_fields(data: dict) -> None:  # type: ignore[type-arg]
    for ex in data["examples"]:
        assert {"question", "answer", "citations", "grounding", "timing_ms", "passages", "model"} <= set(ex)
        assert ex["retriever"].startswith("local/")


def test_notice_shipped_with_site() -> None:
    assert (ROOT / "site" / "THIRD_PARTY_NOTICES.txt").read_text() == (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "THIRD_PARTY_NOTICES.txt" in (ROOT / "site" / "index.html").read_text()
