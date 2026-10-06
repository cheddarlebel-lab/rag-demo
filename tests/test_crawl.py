from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_demo import config
from rag_demo.crawl import CrawlRefused, CrawlStopped, Fetcher, crawl, in_scope, load_robots
from tests.conftest import FakeResponse

PREFIX = "https://docs.test/en/stable/"
HTML = '<html><body><a href="page2/">p2</a><a href="https://evil.test/x/">x</a></body></html>'


class FakeSession:
    """Maps URL -> list of responses (consumed in order; last one repeats)."""

    def __init__(self, routes: dict[str, list[FakeResponse]]):
        self.routes = routes
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, bool]] = []

    def get(self, url: str, timeout: float = 0, allow_redirects: bool = True) -> FakeResponse:
        self.calls.append((url, allow_redirects))
        queue = self.routes.get(url)
        if not queue:
            return FakeResponse(404, text="")
        return queue.pop(0) if len(queue) > 1 else queue[0]


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def html(body: str = HTML) -> FakeResponse:
    return FakeResponse(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


def redirect(to: str) -> FakeResponse:
    return FakeResponse(301, text="", headers={"Location": to})


def make(routes: dict[str, list[FakeResponse]]) -> tuple[Fetcher, FakeSession, Clock]:
    s, c = FakeSession(routes), Clock()
    return Fetcher(session=s, delay=0, sleep=c.sleep, clock=c.now), s, c  # type: ignore[arg-type]


ROBOTS_OK = FakeResponse(200, text="User-agent: *\nDisallow: /en/stable/private/\n")


@pytest.mark.parametrize("url,ok", [
    (PREFIX, True), (PREFIX + "a/b/", True), ("https://docs.test/en/stable", True),
    ("https://docs.test/en/stable-old/", False), ("http://docs.test/en/stable/", False),
    ("https://docs.test.evil.test/en/stable/", False), ("https://user:pw@docs.test/en/stable/", False),
    ("ftp://docs.test/en/stable/", False), ("https://docs.test/en/", False),
    ("https://docs.test/en/stable/../../private/", False), ("https://docs.test/en/stable/%2e%2e/x/", False),
    ("https://docs.test/en/stable/a/./b/", False), ("https://docs.test/en/stable/..%2fprivate", False),
    ("https://docs.test/en/stable\\..\\private", False),
])
def test_in_scope(url: str, ok: bool) -> None:
    assert in_scope(url, PREFIX) is ok


def test_scope_judges_the_url_requests_will_send() -> None:
    import requests
    sneaky = "https://docs.test/en/stable/../../private/"
    sent = requests.Request("GET", sneaky).prepare().url
    assert sent == "https://docs.test/private/"          # what would actually be requested
    assert not in_scope(sneaky, PREFIX) and not in_scope(sent or "", PREFIX)


def test_delay_floor_cannot_be_zero() -> None:
    f, _, _ = make({})
    assert f.delay == config.MIN_CRAWL_DELAY_S >= 1.0


def test_seed_outside_prefix_rejected(tmp_path: Path) -> None:
    f, _, _ = make({})
    with pytest.raises(ValueError, match="outside the allowed prefix"):
        crawl("https://other.test/", PREFIX, fetcher=f, raw_dir=tmp_path)


@pytest.mark.parametrize("status", [401, 403])
def test_robots_auth_errors_refuse(status: int) -> None:
    f, _, _ = make({"https://docs.test/robots.txt": [FakeResponse(status)]})
    with pytest.raises(CrawlRefused):
        load_robots(PREFIX, f)


@pytest.mark.parametrize("status", [404, 410])
def test_robots_missing_allows(status: int) -> None:
    f, _, _ = make({"https://docs.test/robots.txt": [FakeResponse(status)]})
    assert load_robots(PREFIX, f).can_fetch(config.USER_AGENT, PREFIX + "x/")


@pytest.mark.parametrize("status", [429, 503])
def test_robots_overload_backs_off_then_stops(status: int) -> None:
    f, s, c = make({"https://docs.test/robots.txt": [FakeResponse(status)]})
    with pytest.raises(CrawlStopped):
        load_robots(PREFIX, f)
    assert len(s.calls) == config.MAX_RETRIES + 1
    assert sum(1 for x in c.sleeps if x >= 2) >= config.MAX_RETRIES  # exponential backoff happened


def test_robots_retry_after_then_success() -> None:
    f, _, c = make({"https://docs.test/robots.txt": [FakeResponse(429, headers={"Retry-After": "7"}), ROBOTS_OK]})
    rp = load_robots(PREFIX, f)
    assert 7 in c.sleeps
    assert not rp.can_fetch(config.USER_AGENT, PREFIX + "private/x/")


def test_robots_offsite_redirect_refused() -> None:
    f, _, _ = make({"https://docs.test/robots.txt": [redirect("https://cdn.test/robots.txt")]})
    with pytest.raises(CrawlRefused):
        load_robots(PREFIX, f)


def test_crawl_paces_every_request_and_never_auto_redirects(tmp_path: Path) -> None:
    f, s, c = make({"https://docs.test/robots.txt": [ROBOTS_OK], PREFIX: [html()],
                    PREFIX + "page2/": [html("<html></html>")]})
    pages = crawl(PREFIX, PREFIX, fetcher=f, raw_dir=tmp_path)
    assert [p["url"] for p in pages] == [PREFIX, PREFIX + "page2/"]
    assert all(not ar for _, ar in s.calls)              # allow_redirects=False everywhere
    assert len(c.sleeps) == len(s.calls) - 1 and all(x >= 1.0 for x in c.sleeps)  # robots fetch included
    assert not any("evil.test" in u for u, _ in s.calls)  # off-site link never requested


def test_redirect_out_of_scope_is_skipped(tmp_path: Path) -> None:
    f, s, _ = make({"https://docs.test/robots.txt": [ROBOTS_OK], PREFIX: [redirect("https://evil.test/")]})
    assert crawl(PREFIX, PREFIX, fetcher=f, raw_dir=tmp_path) == []
    assert not any("evil.test" in u for u, _ in s.calls)


def test_redirect_to_robots_disallowed_is_skipped(tmp_path: Path) -> None:
    f, s, _ = make({"https://docs.test/robots.txt": [ROBOTS_OK], PREFIX: [redirect(PREFIX + "private/a/")]})
    assert crawl(PREFIX, PREFIX, fetcher=f, raw_dir=tmp_path) == []
    assert not any("private" in u for u, _ in s.calls)


def test_redirect_in_scope_records_final_url(tmp_path: Path) -> None:
    f, _, _ = make({"https://docs.test/robots.txt": [ROBOTS_OK], PREFIX: [redirect(PREFIX + "home/")],
                    PREFIX + "home/": [html("<html></html>")]})
    pages = crawl(PREFIX, PREFIX, fetcher=f, raw_dir=tmp_path)
    assert pages == [{"url": PREFIX + "home/", "file": pages[0]["file"], "requested": PREFIX}]


def test_redirect_limit(tmp_path: Path) -> None:
    routes: dict[str, list[FakeResponse]] = {"https://docs.test/robots.txt": [ROBOTS_OK]}
    for i in range(10):
        routes[PREFIX + (f"r{i}/" if i else "")] = [redirect(PREFIX + f"r{i + 1}/")]
    f, s, _ = make(routes)
    assert crawl(PREFIX, PREFIX, fetcher=f, raw_dir=tmp_path) == []
    assert len(s.calls) == 1 + config.MAX_REDIRECTS + 1


def test_page_overload_stops_crawl(tmp_path: Path) -> None:
    f, _, _ = make({"https://docs.test/robots.txt": [ROBOTS_OK], PREFIX: [FakeResponse(503)]})
    with pytest.raises(CrawlStopped):
        crawl(PREFIX, PREFIX, fetcher=f, raw_dir=tmp_path)


def test_cache_is_reused(tmp_path: Path) -> None:
    routes: dict[str, Any] = {"https://docs.test/robots.txt": [ROBOTS_OK], PREFIX: [html("<html></html>")]}
    f, _, _ = make(routes)
    crawl(PREFIX, PREFIX, fetcher=f, raw_dir=tmp_path)
    f2, s2, _ = make({"https://docs.test/robots.txt": [ROBOTS_OK]})
    assert [p["url"] for p in crawl(PREFIX, PREFIX, fetcher=f2, raw_dir=tmp_path)] == [PREFIX]
    assert [u for u, _ in s2.calls] == ["https://docs.test/robots.txt"]
