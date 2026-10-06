from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rag_demo.__main__ import main, positive_int
from rag_demo.manifest import IncompatibleArtifacts, require_compatible


def test_require_compatible() -> None:
    require_compatible("x", {"a": 1}, {"a": 1, "b": 2}, ("a",))
    with pytest.raises(IncompatibleArtifacts, match="corpus_fingerprint"):
        require_compatible("x", {"corpus_fingerprint": "old"}, {"corpus_fingerprint": "new"}, ("corpus_fingerprint",))


def _ingest(**over: object) -> dict:  # type: ignore[type-arg]
    base = {"status": "complete", "corpus_fingerprint": "c", "workspace": "ws", "anythingllm_version": "1",
            "workspace_content_fingerprint": "w", "expected_content_fingerprint": "w", "instance_settings": {"A": 1},
            "workspace_settings": {"topN": 5}, "foreign_documents": 0}
    base.update(over)
    return base


CURRENT = {"corpus_fingerprint": "c", "questions_fingerprint": "q", "llm": "ollama/m"}
GOOD_MANIFEST = {"corpus_fingerprint": "c", "workspace": "ws", "anythingllm_version": "1",
                 "workspace_content_fingerprint": "w", "instance_settings": {"A": 1},
                 "workspace_settings": {"topN": 5}, "foreign_documents": 0}


def test_anythingllm_artifact_checks() -> None:
    from rag_demo.export import check_anythingllm_artifacts as check
    check(_ingest(), {"manifest": {**GOOD_MANIFEST, "questions_fingerprint": "q"}},
          {"manifest": {**GOOD_MANIFEST, "analysis_llm": "ollama/m"}}, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="status"):
        check(_ingest(status="incomplete"), None, None, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="workspace content"):
        check(_ingest(workspace_content_fingerprint="stale"), None, None, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="instance_settings"):
        check(_ingest(), {"manifest": {**GOOD_MANIFEST, "questions_fingerprint": "q", "instance_settings": {}}},
              None, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="analysis_llm"):
        check(_ingest(), None, {"manifest": {**GOOD_MANIFEST, "analysis_llm": "anthropic/x"}}, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="missing"):
        check(None, {"manifest": {}}, None, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="workspace_settings"):
        check(_ingest(), None, {"manifest": {**GOOD_MANIFEST, "analysis_llm": "ollama/m",
                                             "workspace_settings": {"topN": 9}}}, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="does not own"):
        check(_ingest(foreign_documents=1), None, None, CURRENT)
    with pytest.raises(IncompatibleArtifacts, match="rerun"):
        check(_ingest(), None, None, CURRENT, rerun=_ingest(status="failed"))
    with pytest.raises(IncompatibleArtifacts, match="rerun"):
        check(_ingest(), None, None, CURRENT, rerun=_ingest(anythingllm_version="2"))
    check(_ingest(), None, None, CURRENT, rerun=_ingest())


@pytest.mark.parametrize("v", ["0", "-3", "x"])
def test_positive_int_rejects(v: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(v)


def test_cli_rejects_empty_ks(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["eval", "--ks", ","])
    assert "comma-separated" in capsys.readouterr().err


def test_cli_rejects_k_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["ask", "q", "-k", "0"])
    assert "positive integer" in capsys.readouterr().err


def test_export_creates_output_dir(tmp_path: Path) -> None:
    """Regression for the Docker bug: export must create site/ when it does not exist."""
    from rag_demo.export import write_json
    target = tmp_path / "app" / "site" / "data.json"
    write_json(target, {"ok": True})
    assert target.read_text().strip().startswith("{")


def _row(trap: bool = False, error: bool = False, **kw: object) -> dict:  # type: ignore[type-arg]
    if error:
        return {"q": "x", "trap": trap, "error": "schema"}
    base = {"q": "x", "trap": trap, "citations_ref_valid": 1, "markers_valid": True, "cites_target": True,
            "insufficient_context": trap, "quotes_given": 1, "quotes_verified": 1, "llm_ms": 10}
    base.update(kw)
    return base


def test_answer_eval_counts_errors_as_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_demo import answer_eval
    monkeypatch.setattr(answer_eval, "run_manifest", lambda llm: {"llm": llm})
    s = answer_eval.summarize([_row(), _row(error=True), _row(trap=True), _row(trap=True, error=True)],
                              "fake/m", "subset-abc", "custom.jsonl", 1, 1.0, "set")
    assert s["questions"] == 2 and s["trap_questions"] == 2 and s["output_errors"] == 2
    assert s["answered_with_citations"] == 0.5 and s["trap_abstain_rate"] == 0.5
    assert s["manifest"]["questions_fingerprint"] == "subset-abc" and s["manifest"]["limit"] == 1


def test_median_is_a_true_median(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_demo import answer_eval
    monkeypatch.setattr(answer_eval, "run_manifest", lambda llm: {"llm": llm})
    s = answer_eval.summarize([_row(llm_ms=10), _row(llm_ms=20)], "m", "q", "eval/questions.jsonl", None, 0, "set")
    assert s["median_llm_ms"] == 15


def test_answer_set_fingerprint_covers_traps_and_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rag_demo import answer_eval, config
    base = answer_eval.answer_set_fingerprint()
    monkeypatch.setattr(answer_eval, "TRAPS", [*answer_eval.TRAPS, "Another trap?"])
    assert answer_eval.answer_set_fingerprint() != base
    monkeypatch.undo()
    q = tmp_path / "eval" / "questions.jsonl"
    q.parent.mkdir()
    q.write_text('{"q": "a", "targets": [{"page": "x/"}]}\n')
    monkeypatch.setattr(config, "ROOT", tmp_path)
    one = answer_eval.answer_set_fingerprint()
    q.write_text('{"q": "a", "targets": [{"page": "y/"}]}\n')          # same question, new target label
    assert answer_eval.answer_set_fingerprint() != one


def test_question_set_fingerprint_distinguishes_subsets() -> None:
    from rag_demo.answer_eval import question_set_fingerprint
    from rag_demo.evaluate import questions_fingerprint
    assert question_set_fingerprint("eval/questions.jsonl", None) == questions_fingerprint()
    sub = question_set_fingerprint("eval/questions.jsonl", 3)
    assert sub.startswith("subset-") and sub != question_set_fingerprint("eval/questions.jsonl", 4)
