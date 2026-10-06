"""Polite crawler.

* Reads robots.txt first and refuses any URL it disallows.
* Honors robots.txt Crawl-delay, with a configurable floor (default 1 req/s).
* Stays inside one URL prefix, strips fragments/queries, dedupes.
* Caches raw HTML on disk so re-runs never re-hit the server (use --refresh to force).
* Sends an identifying User-Agent.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from . import config


def url_to_filename(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16] + ".html"


def normalize(url: str) -> str:
    url, _ = urldefrag(url)
    url = url.split("?", 1)[0]
    if url.endswith("/index.html"):
        url = url[: -len("index.html")]
    return url


def load_robots(seed: str, session: requests.Session) -> urllib.robotparser.RobotFileParser:
    parts = urlparse(seed)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    resp = session.get(robots_url, timeout=20)
    if resp.status_code >= 400:
        rp.parse([])  # no robots.txt -> everything allowed
    else:
        rp.parse(resp.text.splitlines())
    return rp


def crawl(seed: str = config.SEED_URL, prefix: str = config.ALLOWED_PREFIX,
          max_pages: int = config.MAX_PAGES, refresh: bool = False) -> list[dict]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.RAW_DIR / "manifest.json"
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT

    rp = load_robots(seed, session)
    delay = max(config.CRAWL_DELAY_S, float(rp.crawl_delay(config.USER_AGENT) or 0))
    print(f"robots.txt loaded; delay={delay}s; max_pages={max_pages}")

    queue, seen, pages = deque([normalize(seed)]), {normalize(seed)}, []
    skipped_robots = 0
    last_fetch = 0.0
    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if not rp.can_fetch(config.USER_AGENT, url):
            skipped_robots += 1
            continue
        path = config.RAW_DIR / url_to_filename(url)
        if path.exists() and not refresh:
            html = path.read_text(encoding="utf-8")
        else:
            wait = delay - (time.monotonic() - last_fetch)
            if wait > 0:
                time.sleep(wait)
            last_fetch = time.monotonic()
            try:
                resp = session.get(url, timeout=30)
            except requests.RequestException as exc:
                print(f"  ! {url}: {exc}")
                continue
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                print(f"  - skip {resp.status_code} {url}")
                continue
            html = resp.text
            path.write_text(html, encoding="utf-8")
            print(f"  + [{len(pages) + 1}] {url}")
        pages.append({"url": url, "file": path.name})

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            nxt = normalize(urljoin(url, a["href"]))
            if nxt.startswith(prefix) and nxt not in seen and not nxt.endswith(
                (".zip", ".pdf", ".png", ".jpg", ".svg", ".txt", ".js", ".css")
            ) and "/_" not in nxt[len(prefix):] \
                    and not nxt[len(prefix):].startswith(config.SKIP_PATHS):
                seen.add(nxt)
                queue.append(nxt)

    manifest_path.write_text(json.dumps(
        {"seed": seed, "prefix": prefix, "user_agent": config.USER_AGENT, "delay_s": delay,
         "skipped_by_robots": skipped_robots, "pages": pages}, indent=2))
    print(f"done: {len(pages)} pages, {skipped_robots} skipped by robots.txt")
    return pages
