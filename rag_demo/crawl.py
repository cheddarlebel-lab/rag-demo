"""Polite crawler.

* Validates the seed and every URL (scheme, host, path prefix) before requesting it.
* Reads robots.txt first. 404/410 (and other non-auth 4xx) mean "no robots.txt";
  401/403 mean the crawl is refused; 429/5xx are retried with backoff, then the
  crawl stops rather than guessing.
* Every request - robots.txt, pages and each redirect hop - goes through one paced
  fetcher, so the delay (>= 1 s, a larger robots Crawl-delay wins) applies to all of them.
* Redirects are followed by hand (max 5 hops); each hop is checked against scope and
  robots.txt, and the final URL is stored as the page's provenance.
* 429 / 5xx on a page: back off (honouring Retry-After, capped) and retry; if it
  persists, stop the crawl.
* Caches raw HTML on disk so re-runs never re-hit the server (use --refresh to force).
* Sends an identifying User-Agent.

Robots permission is not the same as permission under a site's terms of service;
check those yourself before pointing this at another site.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.robotparser
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from . import config
from .log import event, get_logger

log = get_logger(__name__)

SKIP_EXT = (".zip", ".pdf", ".png", ".jpg", ".svg", ".txt", ".js", ".css")


class CrawlRefused(RuntimeError):
    """robots.txt or the server says we may not crawl."""


class CrawlStopped(RuntimeError):
    """The server kept signalling overload (429/5xx); we stop instead of pushing on."""


def url_to_filename(url: str) -> str:
    return hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:16] + ".html"


def canonical(url: str) -> str:
    """The exact URL `requests` will send (it resolves dot segments and percent-encodes).

    Scope and robots.txt checks run on this form, so they judge the request that is
    actually made rather than the string that was found in a link.
    """
    prepared = requests.models.PreparedRequest()
    try:
        prepared.prepare_url(url, None)
    except requests.RequestException:
        return ""
    return prepared.url or ""


def normalize(url: str) -> str:
    url, _ = urldefrag(url)
    url = url.split("?", 1)[0]
    url = canonical(url)
    if url.endswith("/index.html"):
        url = url[: -len("index.html")]
    return url


_DOT_SEGMENT = re.compile(r"(^|/)(\.|%2e){1,2}(/|$)", re.I)


def in_scope(url: str, prefix: str) -> bool:
    """Parsed comparison (not str.startswith) so 'https://host.evil' or '/en/stable-x' can't sneak in.

    The URL is canonicalized first; any remaining (possibly encoded) dot segment or a
    backslash is rejected outright.
    """
    raw = urlparse(url)
    if "\\" in url or _DOT_SEGMENT.search(raw.path) or "%2f" in raw.path.lower():
        return False
    u, p = urlparse(canonical(url)), urlparse(prefix)
    if u.scheme not in ("http", "https") or u.scheme != p.scheme:
        return False
    if (u.hostname or "").lower() != (p.hostname or "").lower() or u.port != p.port:
        return False
    if u.username or u.password:
        return False
    ppath = p.path if p.path.endswith("/") else p.path + "/"
    return u.path in (p.path, p.path.rstrip("/")) or u.path.startswith(ppath)


class Fetcher:
    """One paced HTTP client: every request waits until `delay` seconds after the previous one."""

    def __init__(self, session: requests.Session | None = None, delay: float = config.CRAWL_DELAY_S,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.delay = max(config.MIN_CRAWL_DELAY_S, delay)
        self.sleep, self.clock = sleep, clock
        self._last: float | None = None
        self.requests = 0

    def get(self, url: str, timeout: float = 30) -> requests.Response:
        if self._last is not None:
            wait = self.delay - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        try:
            return self.session.get(url, timeout=timeout, allow_redirects=False)
        finally:
            self._last = self.clock()
            self.requests += 1

    def get_with_backoff(self, url: str) -> requests.Response:
        """GET; on 429/5xx back off and retry up to MAX_RETRIES times, then raise CrawlStopped."""
        for attempt in range(config.MAX_RETRIES + 1):
            resp = self.get(url)
            if resp.status_code != 429 and resp.status_code < 500:
                return resp
            if attempt == config.MAX_RETRIES:
                break
            retry_after = resp.headers.get("Retry-After", "")
            backoff = float(retry_after) if retry_after.isdigit() else self.delay * 2 ** (attempt + 1)
            event(log, "crawl.backoff", logging.WARNING, url=url, status=resp.status_code,
                  wait_s=round(min(backoff, 60), 1))
            self.sleep(min(backoff, 60))
        raise CrawlStopped(f"{url} kept returning {resp.status_code}; stopping the crawl")


def load_robots(seed: str, fetcher: Fetcher) -> urllib.robotparser.RobotFileParser:
    parts = urlparse(seed)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    url = robots_url
    for _ in range(config.MAX_REDIRECTS + 1):
        resp = fetcher.get_with_backoff(url)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            nxt = urljoin(url, resp.headers["Location"])
            n = urlparse(nxt)
            if (n.scheme, n.netloc) != (parts.scheme, parts.netloc):
                raise CrawlRefused(f"robots.txt redirects off-site ({nxt}); refusing to crawl")
            url = nxt
            continue
        break
    else:
        raise CrawlRefused("robots.txt redirect loop; refusing to crawl")
    if resp.status_code in (401, 403):
        raise CrawlRefused(f"robots.txt returned {resp.status_code}; the site does not allow crawling")
    if resp.status_code == 200:
        rp.parse(resp.text.splitlines())
    elif 400 <= resp.status_code < 500:
        rp.parse([])  # 404/410 etc.: no robots.txt -> no restrictions (RFC 9309 "unavailable")
    else:
        raise CrawlRefused(f"unexpected robots.txt status {resp.status_code}; refusing to crawl")
    return rp


def fetch_page(url: str, fetcher: Fetcher, rp: urllib.robotparser.RobotFileParser,
               prefix: str) -> tuple[str | None, str | None, str]:
    """Follow redirects by hand. Returns (final_url, html, reason). html is None when skipped."""
    for _ in range(config.MAX_REDIRECTS + 1):
        if not in_scope(url, prefix):
            return None, None, f"out of scope: {url}"
        if not rp.can_fetch(config.USER_AGENT, url):
            return None, None, f"disallowed by robots.txt: {url}"
        resp = fetcher.get_with_backoff(url)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                return None, None, f"redirect without Location: {url}"
            url = normalize(urljoin(url, loc))
            continue
        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            return None, None, f"status {resp.status_code}: {url}"
        return url, resp.text, "ok"
    return None, None, f"too many redirects: {url}"


def _links(html: str, base: str, prefix: str) -> Iterator[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        nxt = normalize(urljoin(base, str(a["href"])))
        rest = urlparse(nxt).path[len(urlparse(prefix).path):]
        if in_scope(nxt, prefix) and not nxt.endswith(SKIP_EXT) and "/_" not in "/" + rest \
                and not rest.startswith(config.SKIP_PATHS):
            yield nxt


def crawl(seed: str = config.SEED_URL, prefix: str = config.ALLOWED_PREFIX,
          max_pages: int = config.MAX_PAGES, refresh: bool = False,
          fetcher: Fetcher | None = None, raw_dir: Path | None = None) -> list[dict[str, Any]]:
    seed = normalize(seed)
    if not in_scope(seed, prefix):
        raise ValueError(f"seed {seed!r} is outside the allowed prefix {prefix!r}")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    raw_dir = raw_dir or config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or Fetcher()

    rp = load_robots(seed, fetcher)
    fetcher.delay = max(fetcher.delay, float(rp.crawl_delay(config.USER_AGENT) or 0))
    event(log, "crawl.robots_loaded", delay_s=fetcher.delay, max_pages=max_pages)

    queue, seen = deque([seed]), {seed}
    pages: list[dict[str, Any]] = []
    skipped: list[str] = []
    while queue and len(pages) < max_pages:
        url = queue.popleft()
        path = raw_dir / url_to_filename(url)
        if path.exists() and not refresh and rp.can_fetch(config.USER_AGENT, url):
            final, html = url, path.read_text(encoding="utf-8")
        else:
            try:
                got_url, got_html, reason = fetch_page(url, fetcher, rp, prefix)
            except requests.RequestException as exc:
                skipped.append(f"error {type(exc).__name__}: {url}")
                continue
            if got_html is None or got_url is None:
                skipped.append(reason)
                event(log, "crawl.skip", reason=reason)
                continue
            final, html = got_url, got_html
            if final != url:
                if final in seen and any(p["url"] == final for p in pages):
                    continue  # redirected onto a page we already have
                seen.add(final)
                path = raw_dir / url_to_filename(final)
            path.write_text(html, encoding="utf-8")
            event(log, "crawl.fetched", n=len(pages) + 1, url=final)
        pages.append({"url": final, "file": path.name, **({"requested": url} if final != url else {})})
        for nxt in _links(html, final, prefix):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    manifest = {"seed": seed, "prefix": prefix, "user_agent": config.USER_AGENT, "delay_s": fetcher.delay,
                "crawled_at": time.strftime("%Y-%m-%d"), "requests": fetcher.requests,
                "skipped_by_robots": sum(s.startswith("disallowed") for s in skipped),
                "skipped": skipped, "pages": pages}
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    event(log, "crawl.done", pages=len(pages), skipped=len(skipped),
          skipped_by_robots=manifest["skipped_by_robots"], requests=fetcher.requests)
    return pages
