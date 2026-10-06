from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag_demo import retrieve
from rag_demo.evaluate import first_relevant, metrics
from rag_demo.index import Index, IndexValidationError
from rag_demo.retrieve import BM25, Retriever
from tests.conftest import CHUNKS, write_index


@pytest.fixture
def retriever(tiny_index: Path, monkeypatch: pytest.MonkeyPatch) -> Retriever:
    monkeypatch.setattr(retrieve, "embed_query", lambda q: np.array([0, 0, 1, 0], dtype=np.float32))
    return Retriever(Index(tiny_index))


@pytest.mark.parametrize("k", [0, -1, True, 2.5])
def test_k_validation(retriever: Retriever, k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("q", k=k)  # type: ignore[arg-type]


def test_depth_zero_is_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="depth"):
        retriever.search("q", k=1, depth=0)


def test_unknown_mode(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="unknown retriever mode"):
        retriever.search("q", k=1, mode="vector")


def test_modes_and_depth(retriever: Retriever) -> None:
    assert retriever.search("q", k=1, mode="dense")[0]["anchor"] == "json"
    assert retriever.search("route decorator", k=1, mode="bm25")[0]["anchor"] == "routing"
    assert len(retriever.search("q", k=3, mode="hybrid", depth=1)) == 3   # depth is raised to k


def test_empty_corpus_errors() -> None:
    with pytest.raises(IndexValidationError, match="empty"):
        BM25([])


def test_index_validation(tmp_path: Path) -> None:
    with pytest.raises(IndexValidationError, match="embedding model"):
        Index(write_index(tmp_path / "a", meta={"embed_model": "other-model"}))
    with pytest.raises(IndexValidationError, match="row mismatch"):
        Index(write_index(tmp_path / "b", meta={"chunks": 99}))
    with pytest.raises(IndexValidationError, match="dimension"):
        Index(write_index(tmp_path / "c", meta={"dims": 768}))
    bad = np.eye(3, 4, dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(IndexValidationError, match="non-finite"):
        Index(write_index(tmp_path / "d", vecs=bad))
    with pytest.raises(IndexValidationError, match="empty"):
        Index(write_index(tmp_path / "e", chunks=[], vecs=np.zeros((0, 4), dtype=np.float32)))
    with pytest.raises(IndexValidationError, match="missing"):
        Index(tmp_path / "nope")


def test_same_row_count_corruption_is_detected(tmp_path: Path) -> None:
    p = write_index(tmp_path / "f")
    (p / "chunks.jsonl").write_text((p / "chunks.jsonl").read_text().replace("MAX_CONTENT_LENGTH", "MAX_BODY"))
    with pytest.raises(IndexValidationError, match="fingerprint"):
        Index(p)
    p = write_index(tmp_path / "g")
    np.save(p / "embeddings.npy", np.eye(3, 4, dtype=np.float32)[::-1].copy())
    with pytest.raises(IndexValidationError, match="checksum"):
        Index(p)


def _fake_corpus(raw: Path, text: str) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    html = (f'<html><head><title>P</title></head><body><div class="body" role="main"><section id="s"><h1>P</h1>'
            f"<p>{text}</p></section></div></body></html>")
    (raw / "p.html").write_text(html)
    (raw / "manifest.json").write_text(json.dumps({"prefix": "https://d.test/", "pages": [
        {"url": "https://d.test/p/", "file": "p.html"}]}))


def test_build_publishes_generations_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_demo import index as index_mod
    monkeypatch.setattr(index_mod, "embed_documents",
                        lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2)
    raw, idx = tmp_path / "raw", tmp_path / "index"
    _fake_corpus(raw, "First version of a paragraph that is long enough to be kept.")
    index_mod.build_index(raw, idx)
    assert idx.is_symlink() and "First version" in Index(idx).chunks[0]["text"]
    first_gen = idx.resolve()

    # An interrupted build (embedding fails) must leave the published index untouched.
    def boom(texts: list[str]) -> None:
        raise RuntimeError("ollama went away")
    monkeypatch.setattr(index_mod, "embed_documents", boom)
    _fake_corpus(raw, "Second version of a paragraph that is long enough to be kept.")
    with pytest.raises(RuntimeError):
        index_mod.build_index(raw, idx)
    assert idx.resolve() == first_gen and "First version" in Index(idx).chunks[0]["text"]

    monkeypatch.setattr(index_mod, "embed_documents",
                        lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2)
    index_mod.build_index(raw, idx)
    assert idx.resolve() != first_gen and "Second version" in Index(idx).chunks[0]["text"]


def test_legacy_directory_is_migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_demo import index as index_mod
    monkeypatch.setattr(index_mod, "embed_documents",
                        lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2)
    raw, idx = tmp_path / "raw", tmp_path / "index"
    write_index(idx)                                    # plain directory, as built by 0.1
    _fake_corpus(raw, "A paragraph that is long enough to be kept as a chunk.")
    index_mod.build_index(raw, idx)
    assert idx.is_symlink() and any(p.name.startswith("index.gen-legacy") for p in tmp_path.iterdir())


def test_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_demo import config
    monkeypatch.setattr(config, "ALLOWED_PREFIX", "https://d.test/docs/")
    labels = [{"q": "a", "targets": [{"page": "config/", "anchors": ["max-content-length"]}]},
              {"q": "b", "targets": [{"page": "errors/", "anchors": ["other"]}]}]
    rankings = [[(c["url"], c["anchor"]) for c in (CHUNKS[1], CHUNKS[0])],
                [(c["url"], c["anchor"]) for c in (CHUNKS[2],)]]
    assert first_relevant(rankings[0], labels[0], "page") == 2
    assert first_relevant(rankings[1], labels[1], "section") is None
    m = metrics(rankings, labels, [1, 3], depth=10)
    assert m["page"] == {"hit@1": 0.5, "hit@3": 1.0, "mrr@10": 0.75}
    assert m["section"] == {"hit@1": 0.0, "hit@3": 0.5, "mrr@10": 0.25}


def test_interrupted_legacy_migration_recovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from rag_demo import index as index_mod
    monkeypatch.setattr(index_mod, "embed_documents",
                        lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2)
    raw, idx = tmp_path / "raw", tmp_path / "index"
    write_index(idx)                                    # legacy plain directory with CHUNKS
    _fake_corpus(raw, "A paragraph that is long enough to be kept as a chunk.")

    def crash(*a: object) -> None:
        raise KeyboardInterrupt  # killed between the legacy rename and the switch
    monkeypatch.setattr(index_mod.os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        index_mod.build_index(raw, idx)
    assert not idx.exists()                              # the gap the review found
    monkeypatch.setattr(index_mod.os, "replace", os.replace)
    loaded = Index(idx)                                  # recover() re-links a complete generation
    assert idx.is_symlink() and len(loaded.chunks) in (1, len(CHUNKS))
