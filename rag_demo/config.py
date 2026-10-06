"""Central configuration. Everything is overridable with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("RAG_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"            # cached HTML, one file per page
INDEX_DIR = DATA / "index"        # chunks.jsonl + embeddings.npy + meta.json

# Corpus: Flask documentation (BSD-3-Clause, Copyright 2010 Pallets). See THIRD_PARTY_NOTICES.md.
DEFAULT_SEED = "https://flask.palletsprojects.com/en/stable/"
SEED_URL = os.environ.get("RAG_SEED", DEFAULT_SEED)
ALLOWED_PREFIX = os.environ.get("RAG_PREFIX", DEFAULT_SEED)
MAX_PAGES = int(os.environ.get("RAG_MAX_PAGES", "120"))

# Politeness: the delay between ANY two requests (robots.txt, pages, redirect hops).
# MIN_CRAWL_DELAY_S is a hard floor that the env var cannot lower; a larger robots.txt
# Crawl-delay always wins.
MIN_CRAWL_DELAY_S = 1.0
CRAWL_DELAY_S = max(MIN_CRAWL_DELAY_S, float(os.environ.get("RAG_CRAWL_DELAY", "1.0")))
MAX_REDIRECTS = 5
MAX_RETRIES = 3                   # for 429 / 5xx, with exponential backoff, then the crawl stops
USER_AGENT = os.environ.get(
    "RAG_USER_AGENT",
    "rag-demo/0.2 (+https://github.com/cheddarlebel-lab/rag-demo; polite research crawler)",
)

# Corpus metadata recorded in exports. Required explicitly for any non-default corpus.
CORPUS_NAME: str | None
CORPUS_LICENSE: str | None
if SEED_URL == DEFAULT_SEED and ALLOWED_PREFIX == DEFAULT_SEED:
    CORPUS_NAME = os.environ.get("RAG_CORPUS_NAME", "Flask documentation (stable, 3.1.x)")
    CORPUS_LICENSE = os.environ.get("RAG_CORPUS_LICENSE", "BSD-3-Clause, Copyright 2010 Pallets")
else:
    CORPUS_NAME = os.environ.get("RAG_CORPUS_NAME")      # None -> export refuses
    CORPUS_LICENSE = os.environ.get("RAG_CORPUS_LICENSE")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
# gemma2:9b is the model all published numbers use. (Informal observation, not a published
# measurement: llama3.1:8b and qwen2.5-coder:7b often left the citations array empty.)
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gemma2:9b")
LLM_PROVIDER = os.environ.get("RAG_LLM", "ollama")  # "ollama" | "claude"
CLAUDE_MODEL = os.environ.get("RAG_CLAUDE_MODEL")   # required when RAG_LLM=claude; no default

CHUNK_MAX_CHARS = int(os.environ.get("RAG_CHUNK_MAX", "1500"))
RETRIEVERS = ("hybrid", "dense", "bm25")
RETRIEVER = os.environ.get("RAG_RETRIEVER", "hybrid")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
# How many candidates each retriever ranks before fusion / truncation. MRR is reported @ this depth.
CANDIDATE_DEPTH = int(os.environ.get("RAG_CANDIDATE_DEPTH", "50"))

# Sphinx-generated index pages are link lists, not content: skip them when crawling/indexing.
SKIP_PATHS = ("genindex/", "py-modindex/", "search/")

# AnythingLLM (optional). Never put the key in a file inside the repo.
ANYTHINGLLM_BASE_URL = os.environ.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001").rstrip("/")
ANYTHINGLLM_WORKSPACE = os.environ.get("ANYTHINGLLM_WORKSPACE", "flask-docs")
ANYTHINGLLM_TIMEOUT_S = float(os.environ.get("ANYTHINGLLM_TIMEOUT", "120"))
