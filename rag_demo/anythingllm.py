"""AnythingLLM integration: ingest the chunked corpus into a workspace, retrieve from it,
and hand its sources to this repo's LLM + citation-validation layer.

Routes used (AnythingLLM Developer API, checked against the instance's /api/docs):

  GET    /api/v1/auth                              - verify the API key
  GET    /api/v1/workspace/{slug}                  - read workspace + embedded documents
  POST   /api/v1/workspace/new                     - create the workspace (chatMode=query)
  GET    /api/v1/documents/folder/custom-documents - list uploaded documents (duplicate check)
  POST   /api/v1/document/raw-text                 - upload one chunk as a raw-text document
  POST   /api/v1/workspace/{slug}/update-embeddings - attach (adds) / detach (deletes) documents
  DELETE /api/v1/system/remove-documents           - delete replaced/orphaned documents
  POST   /api/v1/workspace/{slug}/vector-search    - retrieval only (used by the eval)
  POST   /api/v1/workspace/{slug}/chat             - query-mode chat; its `sources` feed `ask`
  GET    /api/v1/workspaces                        - other workspaces (never delete a doc they embed)
  GET    /api/v1/system/vector-count               - sanity check after ingest
  GET    /api/v1/system                            - effective embedder / vector DB / LLM (allow-listed keys)

Each repo chunk becomes one AnythingLLM document. Metadata: title = breadcrumb,
url/docSource = deep link, chunkSource = "rag-demo-chunk:<chunk id>", docAuthor and
description carry the corpus license and the ownership marker below.

Duplicate policy (makes reruns safe). Every document carries the marker
"rag-demo:<workspace>:<chunk key>:<content hash>"; only documents whose marker names
this workspace are ever touched. Unchanged chunks are skipped. A changed chunk is
uploaded and embedded first, and its old document is retired only after every new
copy is verified in the workspace (a failed upload keeps the old one). Owned documents
for chunks that no longer exist are detached and deleted (--no-prune keeps them).
Nothing is deleted while another workspace still embeds it.

AnythingLLM may re-split long documents with its own text splitter; with the
instance's default splitter length (the embedder's max chunk length) the <=1500-char
chunks here are not split, and `ingest` reports the vector count so this can be checked.

The API key is read from ANYTHINGLLM_API_KEY only. It is never logged, never written
to artifacts, and is stripped from error messages.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from . import config
from .evaluate import Hit, first_relevant, load_labels, metrics, questions_fingerprint
from .llm import LLM, get_llm
from .log import event, get_logger
from .pipeline import SCHEMA, analyze, validate_output

log = get_logger(__name__)

MARKER = re.compile(r"rag-demo:(?P<ws>[a-z0-9-]+):(?P<key>[0-9a-f]{16}):(?P<hash>[0-9a-f]{12})")
CHUNK_SOURCE_PREFIX = "rag-demo-chunk:"
DOC_FOLDER = "custom-documents"
_META_HEADER = re.compile(r"^<document_metadata>.*?</document_metadata>\s*", re.S)


class AnythingLLMError(RuntimeError):
    pass


def chunk_key(chunk_id: str) -> str:
    return hashlib.sha1(chunk_id.encode(), usedforsecurity=False).hexdigest()[:16]


def content_hash(chunk: dict[str, Any]) -> str:
    return hashlib.sha1(f"{chunk['breadcrumb']}\n{chunk['text']}\n{chunk['link']}".encode(),
                        usedforsecurity=False).hexdigest()[:12]


def marker(slug: str, chunk: dict[str, Any]) -> str:
    """Ownership marker: this workspace's copy of this chunk at this content version."""
    return f"rag-demo:{slug}:{chunk_key(chunk['id'])}:{content_hash(chunk)}"


def document_payload(chunk: dict[str, Any], license_: str, slug: str) -> dict[str, Any]:
    """The raw-text upload body for one chunk. The breadcrumb leads the text, as in the local index."""
    return {
        "textContent": f"{chunk['breadcrumb']}\n\n{chunk['text']}",
        "metadata": {
            "title": chunk["breadcrumb"],
            "url": chunk["link"],
            "docSource": chunk["link"],
            "chunkSource": CHUNK_SOURCE_PREFIX + chunk["id"],
            "docAuthor": license_,
            "description": f"{marker(slug, chunk)} | {license_}",
        },
    }


class AnythingLLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None, session: requests.Session | None = None):
        self.base_url = (base_url or config.ANYTHINGLLM_BASE_URL).rstrip("/")
        key = api_key if api_key is not None else os.environ.get("ANYTHINGLLM_API_KEY", "")
        if not key:
            raise AnythingLLMError(
                "ANYTHINGLLM_API_KEY is not set. Create a key in AnythingLLM (Settings > Developer API) "
                "and export it; never commit it.")
        self._key = key
        self.timeout = timeout or config.ANYTHINGLLM_TIMEOUT_S
        self.session = session or requests.Session()

    def _redact(self, s: str) -> str:
        return s.replace(self._key, "<redacted>") if self._key else s

    def _req(self, method: str, path: str, *, json_body: Any = None, params: dict[str, Any] | None = None,
             timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/api{path}"
        try:
            resp = self.session.request(method, url, json=json_body, params=params,
                                        timeout=timeout or self.timeout, allow_redirects=False,
                                        headers={"Authorization": f"Bearer {self._key}",
                                                 "Accept": "application/json"})
        except requests.RequestException as exc:
            raise AnythingLLMError(self._redact(f"{method} {path} failed: {type(exc).__name__}: {exc}")) from None
        if 300 <= resp.status_code < 400:
            raise AnythingLLMError(f"{method} {path}: unexpected redirect (HTTP {resp.status_code}); "
                                   "refusing to follow it with the API key")
        if resp.status_code in (401, 403):
            raise AnythingLLMError(f"{method} {path}: HTTP {resp.status_code} - the API key was rejected")
        if resp.status_code >= 400:
            raise AnythingLLMError(self._redact(f"{method} {path}: HTTP {resp.status_code}: {resp.text[:300]}"))
        try:
            body = resp.json()
        except ValueError:
            raise AnythingLLMError(f"{method} {path}: non-JSON response (HTTP {resp.status_code})") from None
        if isinstance(body, dict) and body.get("error") not in (None, False, "null", ""):
            raise AnythingLLMError(self._redact(f"{method} {path}: {body['error']}"))
        return body if isinstance(body, dict) else {"data": body}

    # --- thin route wrappers -------------------------------------------------------
    def check_auth(self) -> None:
        if not self._req("GET", "/v1/auth").get("authenticated"):
            raise AnythingLLMError("GET /v1/auth: key not authenticated")

    def get_workspace(self, slug: str) -> dict[str, Any] | None:
        ws = self._req("GET", f"/v1/workspace/{slug}").get("workspace")
        if isinstance(ws, list):  # this route returns a list
            ws = ws[0] if ws else None
        return ws or None

    def create_workspace(self, name: str, top_n: int, similarity_threshold: float) -> dict[str, Any]:
        ws = self._req("POST", "/v1/workspace/new", json_body={
            "name": name, "chatMode": "query", "topN": top_n,
            "similarityThreshold": similarity_threshold, "openAiTemp": 0})["workspace"]
        return dict(ws)

    def list_documents(self, folder: str = DOC_FOLDER) -> list[dict[str, Any]]:
        return list(self._req("GET", f"/v1/documents/folder/{folder}").get("documents", []))

    def upload_raw_text(self, payload: dict[str, Any]) -> str:
        docs = self._req("POST", "/v1/document/raw-text", json_body=payload).get("documents") or []
        if not docs or not docs[0].get("location"):
            raise AnythingLLMError("POST /v1/document/raw-text: no document location returned")
        return str(docs[0]["location"])

    def update_embeddings(self, slug: str, adds: list[str], deletes: list[str]) -> None:
        self._req("POST", f"/v1/workspace/{slug}/update-embeddings",
                  json_body={"adds": adds, "deletes": deletes}, timeout=max(self.timeout, 900))

    def remove_documents(self, names: list[str]) -> None:
        self._req("DELETE", "/v1/system/remove-documents", json_body={"names": names})

    def vector_search(self, slug: str, query: str, top_n: int, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        body = self._req("POST", f"/v1/workspace/{slug}/vector-search",
                         json_body={"query": query, "topN": top_n, "scoreThreshold": score_threshold})
        return list(body.get("results", []))

    def chat(self, slug: str, message: str, session_id: str) -> dict[str, Any]:
        return self._req("POST", f"/v1/workspace/{slug}/chat",
                         json_body={"message": message, "mode": "query", "sessionId": session_id},
                         timeout=max(self.timeout, 600))

    def list_workspaces(self) -> list[dict[str, Any]]:
        return list(self._req("GET", "/v1/workspaces").get("workspaces", []))

    def vector_count(self) -> int | None:
        v = self._req("GET", "/v1/system/vector-count").get("vectorCount")
        return int(v) if v is not None else None


def _doc_location(doc: dict[str, Any], folder: str = DOC_FOLDER) -> str:
    return f"{folder}/{doc['name']}"


def _load_chunks(index_dir: Path | None = None) -> list[dict[str, Any]]:
    path = (index_dir or config.INDEX_DIR) / "chunks.jsonl"
    if not path.exists():
        raise AnythingLLMError("data/index/chunks.jsonl missing; run `python -m rag_demo index` first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _docpaths(ws: dict[str, Any] | None) -> set[str]:
    return {str(d.get("docpath")) for d in (ws or {}).get("documents", []) if d.get("docpath")}


def ingest(client: AnythingLLMClient, slug: str = config.ANYTHINGLLM_WORKSPACE, limit: int | None = None,
           dry_run: bool = False, prune: bool = True, batch: int = 100, top_n: int = 5,
           similarity_threshold: float = 0.0, chunks: list[dict[str, Any]] | None = None,
           out: str | None = "eval/anythingllm_ingest.json") -> dict[str, Any]:
    """Upload + embed chunks into one workspace. Safe to rerun.

    Ownership: only documents whose marker names *this* workspace are considered ours.
    Order: upload new/changed -> embed -> verify they are in the workspace -> only then
    detach and delete the documents they supersede. A failed replacement keeps the old
    document. Documents still embedded in another workspace are detached but never deleted.
    Raises AnythingLLMError (after writing the manifest) if anything was left incomplete.
    """
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise AnythingLLMError(f"workspace slug {slug!r} must be lowercase letters, digits and hyphens")
    if not config.CORPUS_LICENSE:
        raise AnythingLLMError("set RAG_CORPUS_LICENSE for a non-default corpus before ingesting")
    chunks = chunks if chunks is not None else _load_chunks()
    if limit is not None:
        chunks = chunks[:limit]
    t0 = time.time()
    client.check_auth()

    owned: dict[str, tuple[str, str]] = {}   # chunk key -> (content hash, location), this workspace only
    duplicates: list[str] = []
    for doc in client.list_documents():
        m = MARKER.search(doc.get("description") or "")
        if not m or m.group("ws") != slug:
            continue  # not ours (other workspace or not created by rag-demo): never touched
        if m.group("key") in owned:
            duplicates.append(_doc_location(doc))
        else:
            owned[m.group("key")] = (m.group("hash"), _doc_location(doc))

    wanted: dict[str, dict[str, Any]] = {chunk_key(c["id"]): c for c in chunks}
    to_upload = [c for k, c in wanted.items() if owned.get(k, ("",))[0] != content_hash(c)]
    orphaned = [loc for k, (_, loc) in owned.items() if k not in wanted] if prune and limit is None else []
    plan = {"chunks": len(chunks), "already_uploaded_unchanged": len(chunks) - len(to_upload),
            "to_upload": len(to_upload), "to_replace": sum(1 for c in to_upload if chunk_key(c["id"]) in owned),
            "orphaned": len(orphaned), "duplicates": len(duplicates)}
    event(log, "anythingllm.ingest.plan", workspace=slug, dry_run=dry_run, **plan)
    if dry_run:
        return {"dry_run": True, "workspace": slug, **plan}

    locations = {k: loc for k, (_, loc) in owned.items() if k in wanted}
    superseded: list[str] = []
    failures: list[dict[str, str]] = []
    progress: dict[str, Any] = {"phase": "workspace", "uploaded": 0, "embedded_batches": 0, "removed": 0}

    def write_manifest(status: str, extra: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": status, "workspace": slug,
            "anythingllm_version": os.environ.get("ANYTHINGLLM_VERSION", "unknown"),
            "base_url": "<ANYTHINGLLM_BASE_URL>",  # redacted on purpose
            "corpus_fingerprint": _chunks_fingerprint(chunks),
            "expected_content_fingerprint": expected_content_fingerprint(slug, chunks),
            "generated": time.strftime("%Y-%m-%d"), **plan, "progress": progress,
            "upload_failures": len(failures), "wall_clock_s": round(time.time() - t0, 1),
            "duplicate_policy": ("ownership = marker naming this workspace; skip unchanged (same key+hash); "
                                 "replace changed only after the new copy is embedded; delete orphaned/duplicate "
                                 "owned docs unless another workspace still embeds them"),
            **extra, "failures": failures,
        }
        if out:
            (config.ROOT / out).write_text(json.dumps(summary, indent=2))
        return summary

    try:
        ws = client.get_workspace(slug)
        if ws is None:
            ws = client.create_workspace(slug, top_n, similarity_threshold)
            if ws.get("slug") != slug:
                raise AnythingLLMError(f"workspace was created with slug {ws.get('slug')!r}, expected {slug!r}")
            event(log, "anythingllm.workspace.created", workspace=slug)
        progress["phase"] = "upload"
        for i, c in enumerate(to_upload):
            k = chunk_key(c["id"])
            try:
                new_loc = client.upload_raw_text(document_payload(c, config.CORPUS_LICENSE, slug))
            except AnythingLLMError as exc:
                failures.append({"chunk_id": c["id"], "error": str(exc)})
                continue  # the previous version (if any) stays in place
            if k in locations:
                superseded.append(locations[k])
            locations[k] = new_loc
            progress["uploaded"] += 1
            if (i + 1) % 100 == 0 or i + 1 == len(to_upload):
                event(log, "anythingllm.ingest.uploaded", done=i + 1, total=len(to_upload), failures=len(failures))

        progress["phase"] = "embed"
        in_ws = _docpaths(client.get_workspace(slug))
        adds = [loc for loc in locations.values() if loc not in in_ws]
        for j in range(0, len(adds), batch):
            client.update_embeddings(slug, adds[j:j + batch], [])
            progress["embedded_batches"] += 1
            event(log, "anythingllm.ingest.embedded", done=min(j + batch, len(adds)), total=len(adds))

        progress["phase"] = "verify"
        in_ws = _docpaths(client.get_workspace(slug))
        missing = [k for k, loc in locations.items() if loc not in in_ws]
        # Only retire old documents once every replacement is verified in the workspace.
        stale = (superseded + orphaned + duplicates) if not missing else []
        removed: list[str] = []
        kept_shared: list[str] = []
        if stale:
            progress["phase"] = "retire"
            detach = [loc for loc in stale if loc in in_ws]
            if detach:
                client.update_embeddings(slug, [], detach)
            used_elsewhere: set[str] = set()
            for other in client.list_workspaces():
                if other.get("slug") and other["slug"] != slug:
                    used_elsewhere |= _docpaths(client.get_workspace(str(other["slug"])))
            kept_shared = [loc for loc in stale if loc in used_elsewhere]
            removed = [loc for loc in stale if loc not in used_elsewhere]
            if removed:
                client.remove_documents(removed)
                progress["removed"] = len(removed)
        progress["phase"] = "done"
        member = workspace_membership(client, slug)
        live_fp = member["workspace_content_fingerprint"]
        extra = {"embedded_in_workspace": len(locations) - len(missing), "not_embedded": len(missing),
                 "removed_documents": len(removed), "detached_but_kept_shared": len(kept_shared),
                 **member,
                 "vector_count": client.vector_count(), "instance_settings": instance_settings(client),
                 "workspace_settings": _workspace_settings(client.get_workspace(slug)),
                 "not_embedded_chunk_keys": missing}
    except Exception as exc:
        write_manifest("failed", {"error": str(exc)[:500]})
        raise
    complete = (not failures and not missing and extra["foreign_documents"] == 0
                and live_fp == expected_content_fingerprint(slug, chunks))
    summary = write_manifest("complete" if complete else "incomplete", extra)
    event(log, "anythingllm.ingest.done", status=summary["status"], uploaded=progress["uploaded"],
          failures=len(failures), missing=len(missing), workspace_documents=extra["workspace_documents"],
          foreign_documents=extra["foreign_documents"],
          vector_count=extra["vector_count"], wall_clock_s=summary["wall_clock_s"])
    if not complete:
        raise AnythingLLMError(
            f"ingest incomplete: {len(failures)} upload failure(s), {len(missing)} not embedded, "
            f"{extra['foreign_documents']} document(s) in the workspace that this corpus does not own "
            f"(never deleted automatically), workspace content "
            f"{'matches' if live_fp == summary['expected_content_fingerprint'] else 'does not match'} the corpus; "
            f"old documents were kept. Details in {out or 'the returned summary'}")
    return summary


def expected_content_fingerprint(slug: str, chunks: list[dict[str, Any]]) -> str:
    """Fingerprint of the markers a complete ingest of `chunks` into `slug` would leave in the workspace."""
    return hashlib.sha256("\n".join(sorted(marker(slug, c) for c in chunks)).encode()).hexdigest()[:16]


def workspace_membership(client: AnythingLLMClient, slug: str) -> dict[str, Any]:
    """What the workspace actually embeds: a fingerprint of the owned markers, plus counts.

    Any embedded document that this workspace's ingest does not own (another corpus, a manual
    upload) is counted as foreign. Retrieval could return it, so a non-zero foreign count makes
    the workspace non-comparable; such documents are reported, never deleted automatically.
    """
    markers, total = [], 0
    for d in (client.get_workspace(slug) or {}).get("documents", []):
        total += 1
        meta = d.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = {}
        m = MARKER.search((meta or {}).get("description") or "")
        if m and m.group("ws") == slug:
            markers.append(m.group(0))
    return {"workspace_content_fingerprint": hashlib.sha256("\n".join(sorted(markers)).encode()).hexdigest()[:16],
            "workspace_documents": len(markers), "workspace_documents_total": total,
            "foreign_documents": total - len(markers)}


_SETTING_KEYS = ("EmbeddingEngine", "EmbeddingModelPref", "EmbeddingModelMaxChunkLength", "VectorDB",
                 "LLMProvider", "LLMModel")


def instance_settings(client: AnythingLLMClient) -> dict[str, Any]:
    """Effective embedding / vector DB / LLM settings (an allow-list: never copies keys or URLs)."""
    settings = client._req("GET", "/v1/system").get("settings", {})
    return {k: settings.get(k) for k in _SETTING_KEYS}


def _chunks_fingerprint(chunks: list[dict[str, Any]]) -> str:
    """Same value as index.fingerprint() when `chunks` is the whole chunks.jsonl."""
    body = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def source_to_passage(src: dict[str, Any], rank: int) -> dict[str, Any]:
    """Map one AnythingLLM source/search result to the pipeline's passage shape, keeping provenance."""
    meta = src.get("metadata") or src
    text = _META_HEADER.sub("", src.get("text") or "")
    chunk_source = meta.get("chunkSource") or ""
    title = meta.get("title") or ""
    first, _, rest = text.partition("\n\n")
    breadcrumb = first.strip() if rest else title
    body = rest if rest else text
    link = meta.get("docSource") or meta.get("url") or ""
    return {
        "rank": rank, "link": link, "breadcrumb": breadcrumb, "kind": "anythingllm", "text": body.strip(),
        "provenance": {
            "retriever": "anythingllm",
            "anythingllm_doc_id": src.get("id"),
            "score": round(float(src["score"]), 4) if src.get("score") is not None else None,
            "chunk_id": chunk_source.removeprefix(CHUNK_SOURCE_PREFIX)
            if chunk_source.startswith(CHUNK_SOURCE_PREFIX) else None,
        },
    }


def _hit(p: dict[str, Any]) -> Hit:
    url, _, anchor = p["link"].partition("#")
    return url, anchor


def _workspace_settings(ws: dict[str, Any] | None) -> dict[str, Any]:
    """The workspace settings that affect retrieval/answers, plus how many documents it embeds."""
    ws = ws or {}
    keys = ("chatMode", "topN", "similarityThreshold", "vectorSearchMode", "chatProvider", "chatModel")
    return {k: ws.get(k) for k in keys}


def _ask_manifest(client: AnythingLLMClient, slug: str, llm_name: str) -> dict[str, Any]:
    from .index import fingerprint
    chunks = config.INDEX_DIR / "chunks.jsonl"
    return {"corpus_fingerprint": fingerprint(chunks) if chunks.exists() else None, "workspace": slug,
            "anythingllm_version": os.environ.get("ANYTHINGLLM_VERSION", "unknown"),
            **workspace_membership(client, slug),
            "workspace_settings": _workspace_settings(client.get_workspace(slug)),
            "instance_settings": instance_settings(client),
            "analysis_llm": llm_name, "generated": time.strftime("%Y-%m-%d")}


def ask(client: AnythingLLMClient, question: str, slug: str = config.ANYTHINGLLM_WORKSPACE,
        llm: LLM | None = None) -> dict[str, Any]:
    """AnythingLLM query-mode chat -> its sources -> this repo's LLM layer -> cited, validated JSON."""
    t0 = time.perf_counter()
    resp = client.chat(slug, question, session_id="rag-demo")
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
    passages = [source_to_passage(s, i + 1) for i, s in enumerate(resp.get("sources") or [])]
    anythingllm_side = {"text_response": resp.get("textResponse"),
                        "model": (resp.get("metrics") or {}).get("model"),
                        "note": "AnythingLLM's own answer, kept for comparison; not used for citations"}
    retriever = f"anythingllm/{slug}"
    llm = llm or get_llm()
    manifest = _ask_manifest(client, slug, llm.name)
    if not passages:
        empty = validate_output({"answer": "AnythingLLM returned no sources for this question, so there is "
                                 "nothing to answer from.", "key_points": [], "citations": [],
                                 "confidence": "low", "insufficient_context": True})
        grounding = {"citations_total": 0, "citations_ref_valid": 0, "quotes_given": 0, "quotes_verified": 0,
                     "answer_markers": [], "answer_has_markers": False, "markers_valid": False}
        return {"question": question, **empty, "grounding": grounding, "model": llm.name,
                "retriever": retriever, "timing_ms": {"anythingllm_chat": retrieval_ms, "llm": 0},
                "passages": [], "anythingllm_response": anythingllm_side, "manifest": manifest}
    result = analyze(question, passages, llm, retriever, retrieval_ms)
    result["timing_ms"] = {"anythingllm_chat": retrieval_ms, "llm": result["timing_ms"]["llm"]}
    result["anythingllm_response"] = anythingllm_side
    result["manifest"] = manifest
    return result


def to_markdown(r: dict[str, Any]) -> str:
    lines = [f"# {r['question']}", "", r["answer"], ""]
    if r.get("key_points"):
        lines += ["## Key points", "", *[f"- {k}" for k in r["key_points"]], ""]
    lines += ["## Citations", ""]
    for c in r["citations"]:
        status = "matched" if c["quote_verified"] else ("quote not matched" if c["quote"] else "no quote")
        lines.append(f"- [{c['ref']}] [{c['breadcrumb']}]({c['link']}) ({status})" if c["link"]
                     else f"- [{c['ref']}] invalid reference")
        if c["quote"]:
            lines.append(f"  > {c['quote']}")
    lines += ["", f"Retriever: `{r['retriever']}` · Analysis model: `{r['model']}` · "
              f"confidence: {r['confidence']} · insufficient context: {r['insufficient_context']}", ""]
    lines += ["## Sources returned by the retriever", ""]
    for p in r["passages"]:
        prov = p.get("provenance", {})
        score = f" score={prov['score']}" if prov.get("score") is not None else ""
        lines.append(f"{p['rank']}. [{p['breadcrumb']}]({p['link']}){score}")
    return "\n".join(lines) + "\n"


def run_eval(client: AnythingLLMClient, slug: str = config.ANYTHINGLLM_WORKSPACE,
             ks: tuple[int, ...] = (1, 3, 5, 10), out: str = "eval/anythingllm_results.json",
             compare_local: bool = True) -> dict[str, Any]:
    """Page/section hit@k from AnythingLLM's workspace vector-search, next to the local retrievers
    scored at the same depth on the same questions."""
    from .index import Index, fingerprint
    from .retrieve import Retriever

    labels = load_labels()
    depth = max(ks)
    client.check_auth()
    rankings: list[list[Hit]] = []
    per_question = []
    t0 = time.perf_counter()
    for lab in labels:
        results = client.vector_search(slug, lab["q"], top_n=depth)
        hits = [_hit(source_to_passage(s, i + 1)) for i, s in enumerate(results)]
        rankings.append(hits)
        per_question.append({"q": lab["q"], "top5": ["#".join(h).rstrip("#") for h in hits[:5]],
                             "first_page_rank": first_relevant(hits, lab, "page"),
                             "first_section_rank": first_relevant(hits, lab, "section")})
    avg_ms = (time.perf_counter() - t0) * 1000 / len(labels)
    result: dict[str, Any] = {
        "questions": len(labels),
        "manifest": {"corpus_fingerprint": fingerprint(config.INDEX_DIR / "chunks.jsonl"),
                     "questions_fingerprint": questions_fingerprint(),
                     "workspace_settings": _workspace_settings(client.get_workspace(slug)),
                     **workspace_membership(client, slug),
                     "instance_settings": instance_settings(client),
                     "anythingllm_version": os.environ.get("ANYTHINGLLM_VERSION", "unknown"),
                     "workspace": slug, "route": "POST /api/v1/workspace/{slug}/vector-search",
                     "depth": depth, "ks": list(ks), "generated": time.strftime("%Y-%m-%d")},
        "anythingllm": metrics(rankings, labels, list(ks), depth),
        "anythingllm_avg_query_ms": round(avg_ms, 1),
    }
    if compare_local:
        idx = Index()
        retriever = Retriever(idx)
        local: dict[str, list[list[Hit]]] = {"dense": [], "bm25": [], "hybrid": []}
        for lab in labels:
            for name, order in retriever.rankings(lab["q"], depth).items():
                local[name].append([(idx.chunks[i]["url"], idx.chunks[i]["anchor"]) for i in order])
        result["local_same_depth"] = {n: metrics(v, labels, list(ks), depth) for n, v in local.items()}
        for row, lab, *ranked in zip(per_question, labels, *local.values(), strict=True):
            row["local_same_depth"] = {
                name: {"first_page_rank": first_relevant(h, lab, "page"),
                       "first_section_rank": first_relevant(h, lab, "section")}
                for name, h in zip(local, ranked, strict=True)}
    result["per_question"] = per_question
    (config.ROOT / out).write_text(json.dumps(result, indent=2))
    event(log, "anythingllm.eval.done", page=result["anythingllm"]["page"], section=result["anythingllm"]["section"])
    return result


__all__ = [
    "SCHEMA",
    "AnythingLLMClient",
    "AnythingLLMError",
    "ask",
    "document_payload",
    "ingest",
    "run_eval",
    "source_to_passage",
    "to_markdown",
]
