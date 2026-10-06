"""Shared fixtures. Every test is offline: HTTP is replaced by fakes and nothing calls ollama,
AnythingLLM or Anthropic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag_demo import config


class FakeResponse:
    def __init__(self, status: int = 200, body: Any = None, text: str | None = None,
                 headers: dict[str, str] | None = None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")
        self.headers = headers or {}

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test reaches the real network."""
    import requests

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("test tried to use the network")

    monkeypatch.setattr(requests.Session, "request", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "get", boom)


CHUNKS = [
    {"id": "https://d.test/docs/config/#max-content-length::0", "url": "https://d.test/docs/config/",
     "anchor": "max-content-length", "breadcrumb": "Configuration > MAX_CONTENT_LENGTH",
     "text": "MAX_CONTENT_LENGTH limits the size of the request body in bytes.", "kind": "api",
     "link": "https://d.test/docs/config/#max-content-length"},
    {"id": "https://d.test/docs/quickstart/#routing::1", "url": "https://d.test/docs/quickstart/",
     "anchor": "routing", "breadcrumb": "Quickstart > Routing",
     "text": "Use the route() decorator to bind a function to a URL.", "kind": "prose",
     "link": "https://d.test/docs/quickstart/#routing"},
    {"id": "https://d.test/docs/errors/#json::2", "url": "https://d.test/docs/errors/",
     "anchor": "json", "breadcrumb": "Errors > Returning JSON",
     "text": "Register an error handler that returns a JSON response for API errors.", "kind": "prose",
     "link": "https://d.test/docs/errors/#json"},
]


def write_index(path: Path, chunks: list[dict[str, Any]] = CHUNKS, vecs: np.ndarray | None = None,
                meta: dict[str, Any] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    if vecs is None:
        vecs = np.eye(len(chunks), 4, dtype=np.float32)
    np.save(path / "embeddings.npy", vecs)
    from rag_demo.index import file_sha256, fingerprint
    m = {"embed_model": config.EMBED_MODEL, "dims": int(vecs.shape[1]) if vecs.ndim == 2 else 0,
         "chunks": len(chunks), "corpus_fingerprint": fingerprint(path / "chunks.jsonl"),
         "embeddings_sha256": file_sha256(path / "embeddings.npy")}
    m.update(meta or {})
    (path / "meta.json").write_text(json.dumps(m))
    return path


@pytest.fixture
def tiny_index(tmp_path: Path) -> Path:
    return write_index(tmp_path / "index")
