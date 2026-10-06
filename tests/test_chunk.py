from __future__ import annotations

import pytest

from rag_demo.chunk import FENCE, _pack, chunk_page


def balanced(s: str) -> bool:
    return sum(1 for line in s.split("\n") if line.lstrip().startswith(FENCE)) % 2 == 0


def test_single_overlong_line_is_bounded() -> None:
    pieces = _pack(["x" * 3000], 1500)
    assert len(pieces) >= 2 and all(len(p) <= 1500 for p in pieces)
    assert "".join(pieces) == "x" * 3000


def test_oversized_code_block_fragments_are_balanced_and_bounded() -> None:
    code = "```\n" + "\n".join(f"line_{i} = {i} * 2  # some comment text" for i in range(200)) + "\n```"
    pieces = _pack(["intro paragraph", code, "outro"], 500)
    assert all(len(p) <= 500 for p in pieces)
    assert all(balanced(p) for p in pieces)
    assert sum(p.count("line_") for p in pieces) == 200


def test_small_blocks_pack_together() -> None:
    assert _pack(["a" * 10, "b" * 10], 100) == ["a" * 10 + "\n\n" + "b" * 10]


@pytest.mark.parametrize("limit", [0, -5, 10, 1.5])
def test_rejects_bad_limits(limit: object) -> None:
    with pytest.raises(ValueError):
        _pack(["abc"], limit)  # type: ignore[arg-type]


def test_chunk_page_sections_and_breadcrumbs() -> None:
    html = """<html><head><title>Config &#8212; Docs</title></head><body><div class="body" role="main">
      <section id="config"><h1>Config</h1><p>Intro text that is long enough to keep as a chunk.</p>
        <section id="upload"><h2>Uploads</h2><p>Set MAX_CONTENT_LENGTH to limit the request body size.</p>
        <div class="highlight"><pre>app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024</pre></div></section>
      </section></div></body></html>"""
    chunks = chunk_page(html, "https://d.test/config/", limit=200)
    by_anchor = {c.anchor: c for c in chunks}
    assert by_anchor["upload"].breadcrumb == "Config > Uploads"
    assert by_anchor["upload"].link == "https://d.test/config/#upload"
    assert "```" in by_anchor["upload"].text and balanced(by_anchor["upload"].text)
    assert all(len(c.text) <= 200 for c in chunks)
