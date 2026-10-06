"""Build the vector index: cached HTML -> chunks -> embeddings on disk.

The "vector store" is deliberately boring: chunks.jsonl + a float32 .npy matrix.
At this corpus size (~1k chunks) brute-force cosine is exact and takes <1 ms;
swapping in pgvector/Qdrant/LanceDB only means re-implementing `Index.search`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .chunk import chunk_page
from .embed import embed_documents, embed_query


def build_index() -> None:
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text())
    chunks = []
    pages = [p for p in manifest["pages"]
             if not p["url"][len(manifest["prefix"]):].startswith(config.SKIP_PATHS)]
    for page in pages:
        html = (config.RAW_DIR / page["file"]).read_text(encoding="utf-8")
        chunks.extend(chunk_page(html, page["url"]))
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    # the breadcrumb is embedded with the text so headings carry retrieval signal
    vecs = embed_documents([f"{c.breadcrumb}\n\n{c.text}" for c in chunks])
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(config.INDEX_DIR / "embeddings.npy", vecs)
    (config.INDEX_DIR / "meta.json").write_text(json.dumps(
        {"embed_model": config.EMBED_MODEL, "dims": int(vecs.shape[1]), "chunks": len(chunks),
         "pages": len(pages), "chunk_max_chars": config.CHUNK_MAX_CHARS}, indent=2))
    print(f"index written to {config.INDEX_DIR}")


class Index:
    def __init__(self, path: Path = config.INDEX_DIR):
        self.chunks = [json.loads(l) for l in open(path / "chunks.jsonl", encoding="utf-8")]
        self.vecs = np.load(path / "embeddings.npy")

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        q = embed_query(query)
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 4), "rank": r + 1}
                for r, i in enumerate(top)]
