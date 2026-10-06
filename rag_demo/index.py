"""Build and load the vector index: cached HTML -> chunks -> embeddings on disk.

The "vector store" is deliberately boring: chunks.jsonl + a float32 .npy matrix +
meta.json. At this corpus size (~1k chunks) brute-force cosine is exact and takes
<1 ms. Replacing it with pgvector/Qdrant/LanceDB means replacing both `Index`
(storage and loading) and the dense scoring in `retrieve.Retriever`, which reads
`Index.vecs` directly.

Safety:
* every build writes a new, immutable generation directory (`index.gen-<id>/`); the
  `index` path is a symlink that is switched to the new generation with one atomic
  rename (os.replace), so readers see either the old or the new index, never a mix,
  and an interrupted build leaves the previous index untouched. The previous
  generation is kept; older ones are pruned. A pre-0.2 plain `index/` directory is
  migrated once by renaming it to a generation; that one step is not atomic, and if
  it is interrupted, `recover()` (run on every build and load) re-links the newest
  complete generation.
* loading resolves the symlink once, then validates meta.json against the files: the
  chunks fingerprint, the embeddings checksum, row count, dimensions and finite values,
  and refuses an index built with a different embedding model than the one used for queries.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .chunk import chunk_page
from .embed import embed_documents
from .log import event, get_logger

log = get_logger(__name__)


class IndexValidationError(RuntimeError):
    pass


def fingerprint(chunks_path: Path) -> str:
    """sha256 of chunks.jsonl, used to tie eval artifacts to the exact corpus they measured."""
    return hashlib.sha256(chunks_path.read_bytes()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(index_dir: Path) -> bool:
    """If `index_dir` is missing (a legacy migration was interrupted between its two renames),
    point it back at the newest complete generation. Returns True if it repaired anything."""
    if index_dir.exists() or index_dir.is_symlink():
        return False
    gens = sorted((p for p in index_dir.parent.glob(f"{index_dir.name}.gen-*") if (p / "meta.json").exists()),
                  key=lambda p: p.stat().st_mtime)
    if not gens:
        return False
    os.symlink(gens[-1].name, index_dir)
    event(log, "index.recovered", generation=gens[-1].name)
    return True


def _publish(index_dir: Path, gen: Path) -> None:
    """Point `index_dir` (a symlink) at generation `gen` atomically, then prune old generations."""
    link = index_dir.with_name(f".{index_dir.name}.link-{os.getpid()}")
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(gen.name, link)                              # prepared before anything is moved
    if index_dir.exists() and not index_dir.is_symlink():   # one-time migration of a legacy directory
        # Not atomic with the switch below; if interrupted here, recover() restores the legacy generation.
        index_dir.rename(index_dir.with_name(f"{index_dir.name}.gen-legacy-{os.getpid()}"))
    os.replace(link, index_dir)                             # atomic switch
    gens = sorted((p for p in index_dir.parent.glob(f"{index_dir.name}.gen-*") if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    for old in gens[:-2]:                                    # keep current + previous
        if old.resolve() != gen.resolve():
            shutil.rmtree(old, ignore_errors=True)


def build_index(raw_dir: Path | None = None, index_dir: Path | None = None) -> dict[str, Any]:
    raw_dir, index_dir = raw_dir or config.RAW_DIR, index_dir or config.INDEX_DIR
    recover(index_dir)
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    prefix_path = manifest["prefix"]
    pages = [p for p in manifest["pages"] if not p["url"][len(prefix_path):].startswith(config.SKIP_PATHS)]
    chunks = []
    for page in pages:
        html = (raw_dir / page["file"]).read_text(encoding="utf-8")
        chunks.extend(chunk_page(html, page["url"]))
    if not chunks:
        raise IndexValidationError("no chunks were produced; run `python -m rag_demo crawl` first")
    event(log, "index.chunked", pages=len(pages), chunks=len(chunks))
    # the breadcrumb is embedded with the text so headings carry retrieval signal
    vecs = embed_documents([f"{c.breadcrumb}\n\n{c.text}" for c in chunks])

    tmp = index_dir.with_name(f"{index_dir.name}.gen-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True, exist_ok=False)  # always a fresh directory; never reuses a live generation
    with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(tmp / "embeddings.npy", vecs)
    meta = {"embed_model": config.EMBED_MODEL, "dims": int(vecs.shape[1]), "chunks": len(chunks),
            "pages": len(pages), "chunk_max_chars": config.CHUNK_MAX_CHARS,
            "longest_chunk_chars": max(len(c.text) for c in chunks),
            "chunks_with_unbalanced_fences": sum(c.text.count("```") % 2 for c in chunks),
            "corpus_fingerprint": fingerprint(tmp / "chunks.jsonl"),
            "embeddings_sha256": file_sha256(tmp / "embeddings.npy"),
            "crawled_at": manifest.get("crawled_at"), "built_at": time.strftime("%Y-%m-%d")}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    Index(tmp)  # validate before publishing
    _publish(index_dir, tmp)
    event(log, "index.written", path=index_dir.name, **meta)
    return meta


class Index:
    def __init__(self, path: Path | None = None, embed_model: str | None = None):
        path = path or config.INDEX_DIR
        recover(path)
        path = path.resolve()  # resolve the generation symlink exactly once
        embed_model = embed_model or config.EMBED_MODEL
        for name in ("chunks.jsonl", "embeddings.npy", "meta.json"):
            if not (path / name).exists():
                raise IndexValidationError(
                    f"index file {name} missing in {path.name}/; run `python -m rag_demo index`")
        self.meta: dict[str, Any] = json.loads((path / "meta.json").read_text())
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks: list[dict[str, Any]] = [json.loads(line) for line in f if line.strip()]
        self.vecs: np.ndarray = np.load(path / "embeddings.npy")
        self.path = path
        self._validate(embed_model)

    def _validate(self, embed_model: str) -> None:
        m, v = self.meta, self.vecs
        if not self.chunks:
            raise IndexValidationError("index is empty (0 chunks); crawl and index a corpus first")
        if m.get("corpus_fingerprint") != fingerprint(self.path / "chunks.jsonl"):
            raise IndexValidationError("chunks.jsonl does not match meta.json (fingerprint); rebuild the index")
        if m.get("embeddings_sha256") != file_sha256(self.path / "embeddings.npy"):
            raise IndexValidationError("embeddings.npy does not match meta.json (checksum); rebuild the index")
        if m.get("embed_model") != embed_model:
            raise IndexValidationError(
                f"index was built with embedding model {m.get('embed_model')!r} but queries would use "
                f"{embed_model!r}; rebuild the index or set RAG_EMBED_MODEL={m.get('embed_model')}")
        if v.ndim != 2 or v.shape[0] != len(self.chunks) or v.shape[0] != m.get("chunks"):
            raise IndexValidationError(
                f"row mismatch: {len(self.chunks)} chunks, vectors {v.shape}, meta says {m.get('chunks')}")
        if v.shape[1] != m.get("dims"):
            raise IndexValidationError(f"dimension mismatch: vectors have {v.shape[1]}, meta says {m.get('dims')}")
        if not np.all(np.isfinite(v)):
            raise IndexValidationError("index contains non-finite vectors; rebuild it")
