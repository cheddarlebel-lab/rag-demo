"""Central configuration. Everything is overridable with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("RAG_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"            # cached HTML, one file per page
INDEX_DIR = DATA / "index"        # chunks.jsonl + embeddings.npy

# Corpus: Flask documentation (BSD-3-Clause, Copyright 2010 Pallets).
SEED_URL = os.environ.get("RAG_SEED", "https://flask.palletsprojects.com/en/stable/")
ALLOWED_PREFIX = os.environ.get("RAG_PREFIX", "https://flask.palletsprojects.com/en/stable/")
MAX_PAGES = int(os.environ.get("RAG_MAX_PAGES", "120"))
CRAWL_DELAY_S = float(os.environ.get("RAG_CRAWL_DELAY", "1.0"))  # floor; robots.txt Crawl-delay wins if larger
USER_AGENT = os.environ.get(
    "RAG_USER_AGENT",
    "rag-demo/0.1 (+https://github.com/cheddarlebel-lab/rag-demo; polite research crawler)",
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
# gemma2:9b followed the citation schema most reliably of the local models tried
# (llama3.1:8b and qwen2.5-coder:7b often left the citations array empty).
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gemma2:9b")
LLM_PROVIDER = os.environ.get("RAG_LLM", "ollama")  # "ollama" | "claude"

CHUNK_MAX_CHARS = int(os.environ.get("RAG_CHUNK_MAX", "1500"))
RETRIEVER = os.environ.get("RAG_RETRIEVER", "hybrid")  # "hybrid" | "dense" | "bm25"
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# Sphinx-generated index pages are link lists, not content: skip them when crawling/indexing.
SKIP_PATHS = ("genindex/", "py-modindex/", "search/")
