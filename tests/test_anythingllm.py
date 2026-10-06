from __future__ import annotations

import json
import json as json_mod
from typing import Any

import pytest

from rag_demo import anythingllm as al
from rag_demo import config
from tests.conftest import CHUNKS, FakeResponse

KEY = "SECRET-KEY-123"


class FakeServer:
    """In-memory stand-in for the AnythingLLM developer API routes the client uses."""

    def __init__(self, fail_upload_for: str | None = None):
        self.docs: dict[str, dict[str, Any]] = {}       # location -> doc
        self.workspace: dict[str, Any] | None = None
        self.embedded: set[str] = set()
        self.other_ws_docs: set[str] = set()           # embedded by a second workspace "other"
        self.fail_system = False
        self.fail_embed = False
        self.calls: list[tuple[str, str, Any, dict[str, str]]] = []
        self.fail_upload_for = fail_upload_for
        self.n = 0

    def request(self, method: str, url: str, json: Any = None, params: Any = None, timeout: float = 0,
                headers: dict[str, str] | None = None, allow_redirects: bool = True) -> FakeResponse:
        assert allow_redirects is False, "client must not follow redirects with the API key"
        path = url.split("/api", 1)[1]
        self.calls.append((method, path, json, headers or {}))
        if (headers or {}).get("Authorization") != f"Bearer {KEY}":
            return FakeResponse(403, {"error": "No valid api key found."})
        if path == "/v1/auth":
            return FakeResponse(200, {"authenticated": True})
        if path == "/v1/documents/folder/custom-documents":
            docs = [{"name": loc.split("/", 1)[1], **d} for loc, d in self.docs.items()]
            return FakeResponse(200, {"documents": docs})
        if path == "/v1/document/raw-text":
            if self.fail_upload_for and self.fail_upload_for in json["textContent"]:
                return FakeResponse(500, text=f"boom {KEY}")
            self.n += 1
            loc = f"custom-documents/raw-doc-{self.n}.json"
            self.docs[loc] = {"description": json["metadata"]["description"], **json["metadata"]}
            return FakeResponse(200, {"success": True, "error": None, "documents": [{"location": loc}]})
        if path == "/v1/workspaces":
            return FakeResponse(200, {"workspaces": [{"slug": "other"}] + ([self.workspace] if self.workspace else [])})
        if path == "/v1/workspace/other":
            return FakeResponse(200, {"workspace": [{"slug": "other",
                                                     "documents": [{"docpath": p} for p in self.other_ws_docs]}]})
        if path.startswith("/v1/workspace/new"):
            self.workspace = {"slug": json["name"], "documents": []}
            return FakeResponse(200, {"workspace": self.workspace, "message": None})
        if path.endswith("/update-embeddings"):
            if self.fail_embed:
                return FakeResponse(500, text="embedder offline")
            self.embedded |= set(json["adds"])
            self.embedded -= set(json["deletes"])
            return FakeResponse(200, {"workspace": {}})
        if path.startswith("/v1/workspace/"):
            if self.workspace is None:
                return FakeResponse(200, {"workspace": []})
            docs = [{"docpath": p, "metadata": json_mod.dumps(self.docs.get(p, {}))} for p in sorted(self.embedded)]
            return FakeResponse(200, {"workspace": [{**self.workspace, "documents": docs}]})
        if path == "/v1/system/remove-documents":
            for n in json["names"]:
                self.docs.pop(n, None)
            return FakeResponse(200, {"success": True})
        if path == "/v1/system":
            if self.fail_system:
                return FakeResponse(500, text="system down")
            return FakeResponse(200, {"settings": {"EmbeddingEngine": "ollama", "VectorDB": "lancedb",
                                                   "OllamaLLMAuthToken": False, "EmbeddingBasePath": "http://x"}})
        if path == "/v1/system/vector-count":
            return FakeResponse(200, {"vectorCount": len(self.embedded)})
        return FakeResponse(404, text="no route")


def client(server: FakeServer, key: str = KEY) -> al.AnythingLLMClient:
    c = al.AnythingLLMClient("http://alm.test", key)
    c.session.request = server.request  # type: ignore[method-assign]
    return c


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
    with pytest.raises(al.AnythingLLMError, match="ANYTHINGLLM_API_KEY"):
        al.AnythingLLMClient("http://alm.test")


def test_ingest_payload_and_auth(tmp_path: Any) -> None:
    s = FakeServer()
    res = al.ingest(client(s), "flask-docs", chunks=CHUNKS, out=None)
    assert res["status"] == "complete" and res["embedded_in_workspace"] == 3 and res["not_embedded"] == 0
    assert res["workspace_content_fingerprint"] == al.expected_content_fingerprint("flask-docs", CHUNKS)
    assert res["instance_settings"] == {"EmbeddingEngine": "ollama", "VectorDB": "lancedb", "EmbeddingModelPref": None,
                                        "EmbeddingModelMaxChunkLength": None, "LLMProvider": None, "LLMModel": None}
    uploads = [c for c in s.calls if c[1] == "/v1/document/raw-text"]
    meta = uploads[0][2]["metadata"]
    assert meta["title"] == CHUNKS[0]["breadcrumb"] and meta["docSource"] == CHUNKS[0]["link"]
    assert meta["chunkSource"] == "rag-demo-chunk:" + CHUNKS[0]["id"]
    assert config.CORPUS_LICENSE in meta["description"] and "rag-demo:flask-docs:" in meta["description"]
    assert uploads[0][2]["textContent"].startswith(CHUNKS[0]["breadcrumb"] + "\n\n")
    assert all(h["Authorization"] == f"Bearer {KEY}" for *_, h in s.calls)
    assert "base_url" in res and res["base_url"] == "<ANYTHINGLLM_BASE_URL>"
    assert KEY not in json.dumps(res)


def test_rerun_is_idempotent_and_replaces_changed() -> None:
    s = FakeServer()
    al.ingest(client(s), "ws", chunks=CHUNKS, out=None)
    again = al.ingest(client(s), "ws", chunks=CHUNKS, out=None)
    assert again["to_upload"] == 0 and len(s.docs) == 3
    changed = [dict(CHUNKS[0], text="new text"), *CHUNKS[1:2]]   # one changed, one dropped
    res = al.ingest(client(s), "ws", chunks=changed, out=None)
    assert res["to_upload"] == 1 and res["to_replace"] == 1 and res["orphaned"] == 1
    assert len(s.docs) == 2 and s.embedded == set(s.docs)


def test_partial_failure_raises_keeps_old_doc_and_redacts_key(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ROOT", tmp_path)
    s = FakeServer()
    al.ingest(client(s), "ws", chunks=CHUNKS, out=None)
    before = set(s.embedded)
    s.fail_upload_for = "route() decorator"
    changed = [CHUNKS[0], dict(CHUNKS[1], text="Use the route() decorator, now with more words."), CHUNKS[2]]
    with pytest.raises(al.AnythingLLMError, match="ingest incomplete"):
        al.ingest(client(s), "ws", chunks=changed, out="manifest.json")
    assert s.embedded == before and len(s.docs) == 3          # the working document survived
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["upload_failures"] == 1 and manifest["removed_documents"] == 0
    assert KEY not in json.dumps(manifest) and "<redacted>" in manifest["failures"][0]["error"]


def test_embedding_failure_writes_failed_manifest(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ROOT", tmp_path)
    s = FakeServer()
    (tmp_path / "m.json").write_text(json.dumps({"status": "complete"}))   # a stale success record
    s.fail_embed = True
    with pytest.raises(al.AnythingLLMError, match="embedder offline"):
        al.ingest(client(s), "ws", chunks=CHUNKS, out="m.json")
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["status"] == "failed" and m["progress"]["phase"] == "embed" and m["progress"]["uploaded"] == 3


def test_unexpected_slug_writes_failed_manifest(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ROOT", tmp_path)
    s = FakeServer()
    real = s.request

    def renaming(method: str, url: str, **kw: Any) -> FakeResponse:
        resp = real(method, url, **kw)
        if url.endswith("/v1/workspace/new"):
            s.workspace = {"slug": "ws-2", "documents": []}
            return FakeResponse(200, {"workspace": s.workspace})
        return resp

    c = client(s)
    c.session.request = renaming  # type: ignore[method-assign]
    with pytest.raises(al.AnythingLLMError, match="slug"):
        al.ingest(c, "ws", chunks=CHUNKS, out="m.json")
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["status"] == "failed" and m["progress"]["phase"] == "workspace" and m["progress"]["uploaded"] == 0


def test_foreign_document_in_workspace_makes_ingest_incomplete(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ROOT", tmp_path)
    s = FakeServer()
    al.ingest(client(s), "ws", chunks=CHUNKS, out=None)
    s.docs["custom-documents/manual.json"] = {"description": "uploaded by hand"}
    s.embedded.add("custom-documents/manual.json")
    with pytest.raises(al.AnythingLLMError, match="does not own"):
        al.ingest(client(s), "ws", chunks=CHUNKS, out="m.json")
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["status"] == "incomplete" and m["foreign_documents"] == 1 and m["workspace_documents"] == 3
    assert "custom-documents/manual.json" in s.docs                 # reported, never deleted


def test_ownership_is_per_workspace_and_shared_docs_survive() -> None:
    s = FakeServer()
    al.ingest(client(s), "ws", chunks=CHUNKS, out=None)
    foreign = "custom-documents/foreign.json"
    s.docs[foreign] = {"description": al.marker("another-ws", CHUNKS[0])}
    s.other_ws_docs = {next(iter(s.docs))}                  # "other" workspace embeds one of ours
    res = al.ingest(client(s), "ws", chunks=CHUNKS[2:], out=None)   # two chunks dropped -> orphaned
    assert res["orphaned"] == 2
    assert foreign in s.docs                                  # another workspace's marker: untouched
    assert res["detached_but_kept_shared"] == 1 and res["removed_documents"] == 1


def test_redirects_are_refused() -> None:
    s = FakeServer()
    c = client(s)

    def redirecting(method: str, url: str, **kw: Any) -> FakeResponse:
        assert kw["allow_redirects"] is False
        return FakeResponse(302, text="", headers={"Location": "http://elsewhere.test/"})

    c.session.request = redirecting  # type: ignore[method-assign]
    with pytest.raises(al.AnythingLLMError, match="redirect"):
        c.check_auth()


def test_rejected_key_message() -> None:
    with pytest.raises(al.AnythingLLMError, match="API key was rejected"):
        client(FakeServer(), key="wrong").check_auth()


SOURCE = {"id": "abc", "score": 0.81234,
          "text": "<document_metadata>\nsourceDocument: x.txt\n</document_metadata>\n\n"
                  "Configuration > MAX_CONTENT_LENGTH\n\n"
                  "MAX_CONTENT_LENGTH limits the size of the request body in bytes.",
          "title": "configuration-max-content-length.txt", "docSource": CHUNKS[0]["link"],
          "chunkSource": "rag-demo-chunk:" + CHUNKS[0]["id"]}


def test_source_to_passage_keeps_provenance() -> None:
    p = al.source_to_passage(SOURCE, 1)
    assert p["breadcrumb"] == "Configuration > MAX_CONTENT_LENGTH"
    assert p["text"] == CHUNKS[0]["text"] and p["link"] == CHUNKS[0]["link"]
    assert p["provenance"] == {"retriever": "anythingllm", "anythingllm_doc_id": "abc", "score": 0.8123,
                               "chunk_id": CHUNKS[0]["id"]}


def test_ask_routes_sources_into_llm_layer() -> None:
    class C:
        def get_workspace(self, slug: str) -> dict[str, Any]:
            return {"chatMode": "query", "documents": [{"docpath": "x"}]}

        def _req(self, method: str, path: str) -> dict[str, Any]:
            return {"settings": {"VectorDB": "lancedb"}}

        def chat(self, slug: str, q: str, session_id: str) -> dict[str, Any]:
            return {"textResponse": "their answer", "sources": [SOURCE], "metrics": {"model": "gemma2:9b"}}

    class L:
        name = "fake/llm"

        def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
            assert "limits the size of the request body" in user
            return {"answer": "Use it [1].", "key_points": [], "confidence": "high", "insufficient_context": False,
                    "citations": [{"ref": 1, "quote": "limits the size of the request body"}]}

    r = al.ask(C(), "How big?", "ws", llm=L())  # type: ignore[arg-type]
    assert r["retriever"] == "anythingllm/ws" and r["citations"][0]["quote_verified"]
    assert r["passages"][0]["provenance"]["retriever"] == "anythingllm"
    assert r["anythingllm_response"]["text_response"] == "their answer"
    md = al.to_markdown(r)
    assert "anythingllm/ws" in md and "(matched)" in md


def test_ask_without_sources_does_not_call_llm() -> None:
    class C:
        def get_workspace(self, slug: str) -> dict[str, Any]:
            return {"chatMode": "query", "documents": [{"docpath": "x", "metadata": json.dumps(
                {"description": al.marker("ws", CHUNKS[0])})}]}

        def _req(self, method: str, path: str) -> dict[str, Any]:
            return {"settings": {"VectorDB": "lancedb"}}

        def chat(self, slug: str, q: str, session_id: str) -> dict[str, Any]:
            return {"textResponse": "no relevant info", "sources": []}

    class L:
        name = "fake/llm"

        def complete_json(self, *a: Any) -> dict[str, Any]:
            raise AssertionError("should not be called")

    r = al.ask(C(), "q", "ws", llm=L())  # type: ignore[arg-type]
    assert r["insufficient_context"] is True and r["passages"] == []
    assert r["manifest"]["workspace"] == "ws" and r["manifest"]["workspace_documents"] == 1
    assert r["manifest"]["instance_settings"]["VectorDB"] == "lancedb"
