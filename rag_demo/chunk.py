"""Structure-aware chunking for Sphinx-generated HTML.

Instead of slicing text every N characters, we walk the document tree:

* Each <section id=...> becomes a unit with a breadcrumb ("Page > Section > Sub")
  and a deep link (url#anchor) used later as the citation target.
* API reference entries (<dl class="py ...">, e.g. flask.Flask.route) become their
  own units, anchored on the object's id, so a method's docs are never merged
  with the docs of its neighbours.
* Code blocks are kept intact (fenced) and never split across chunks unless a
  single block is itself larger than the chunk budget; then it is split on lines
  (over-long lines at the budget) and each fragment's fence is closed/re-opened.
* Every chunk is at most CHUNK_MAX_CHARS characters (a hard bound).
* Small units are packed paragraph-by-paragraph up to CHUNK_MAX_CHARS; large
  ones are split on block boundaries, each piece re-carrying the breadcrumb.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from . import config

HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
Unit = tuple[list[str], str, str, list[str]]  # (crumbs, anchor, kind, blocks)


def _attr(el: Tag, name: str, default: str = "") -> str:
    v = el.get(name)
    if v is None:
        return default
    return " ".join(v) if isinstance(v, list) else str(v)


def _classes(el: Tag) -> list[str]:
    v = el.get("class")
    if v is None:
        return []
    return [str(c) for c in v] if isinstance(v, list) else str(v).split()


@dataclass
class Chunk:
    id: str
    url: str           # page url
    anchor: str        # section / API object id
    breadcrumb: str    # "Quickstart > Routing > Variable Rules"
    text: str
    kind: str          # "prose" | "api"

    @property
    def link(self) -> str:
        return f"{self.url}#{self.anchor}" if self.anchor else self.url

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["link"] = self.link
        return d


def _clean(s: str) -> str:
    s = s.replace("¶", "").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _block_text(el: Tag) -> str:
    """Render one block element to plain text; code becomes a fenced block."""
    pre = el if el.name == "pre" else el.find("pre")
    if pre is not None and el.name in ("pre", "div"):
        cap = el.find(class_="code-block-caption")
        head = _clean(cap.get_text(" ")) + "\n" if cap else ""
        return head + "```\n" + pre.get_text().rstrip() + "\n```"
    if el.name in ("ul", "ol"):
        items = [_clean(li.get_text(" ")) for li in el.find_all("li", recursive=False)]
        return "\n".join(f"- {i}" for i in items if i)
    if el.name == "table":
        rows = []
        for tr in el.find_all("tr"):
            cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            rows.append(" | ".join(cells))
        return "\n".join(rows)
    return _clean(el.get_text(" "))


def _heading_text(sec: Tag) -> str:
    for child in sec.children:
        if isinstance(child, Tag) and child.name in HEADINGS:
            return _clean(child.get_text(" "))
    return ""


def _units(container: Tag, crumbs: list[str], anchor: str, out: list[Unit]) -> None:
    """Emit (crumbs, anchor, kind, blocks) units, recursing into sections and API entries."""
    blocks: list[str] = []
    nested: list[Unit] = []
    for child in container.children:
        if isinstance(child, NavigableString):
            t = _clean(str(child))
            if t:
                blocks.append(t)
            continue
        if not isinstance(child, Tag) or child.name in HEADINGS:
            continue
        if child.name == "section":
            _units(child, [*crumbs, _heading_text(child)], _attr(child, "id", anchor), nested)
        elif child.name == "dl" and "py" in _classes(child):
            for dt in child.find_all("dt", recursive=False):
                sig = _clean(dt.get_text(" "))
                dd = dt.find_next_sibling("dd")
                obj_id = _attr(dt, "id") or anchor
                name = obj_id.split(".")[-1] if _attr(dt, "id") else sig[:60]
                sub: list[Unit] = []
                if dd is not None:
                    _units(dd, [*crumbs, name], obj_id, sub)
                # the signature leads the first unit of this entry
                if sub and sub[0][1] == obj_id:
                    sub[0] = (sub[0][0], sub[0][1], "api", [sig] + sub[0][3])
                else:
                    sub.insert(0, ([*crumbs, name], obj_id, "api", [sig]))
                nested.extend(sub)
        elif child.name in ("div", "blockquote") and not any(
            c in _classes(child) for c in ("highlight",)) and not any(
            "highlight" in c for c in _classes(child)) and child.find(["section", "dl"]) is not None:
            _units(child, crumbs, anchor, nested)  # transparent wrapper (admonitions etc. with structure)
        else:
            t = _block_text(child)
            if t:
                blocks.append(t)
    if blocks:
        out.append((crumbs, anchor, "prose", blocks))
    out.extend(nested)


FENCE = "```"
_FENCE_OVERHEAD = len(FENCE + "\n") + len("\n" + FENCE)  # reopen + close around a code fragment
MIN_CHUNK_CHARS = 64


def _split_block(block: str, limit: int) -> list[str]:
    """Split one oversized block into pieces of at most `limit` chars.

    Lines longer than the budget are cut at the budget. If a piece starts or ends
    inside a fenced code block, the fence is re-opened / closed so every piece is
    balanced markdown.
    """
    budget = limit - _FENCE_OVERHEAD
    lines: list[str] = []
    for line in block.split("\n"):
        while len(line) > budget:
            lines.append(line[:budget])
            line = line[budget:]
        lines.append(line)
    raw: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        add = len(line) + (1 if buf else 0)
        if buf and size + add > budget:
            raw.append("\n".join(buf))
            buf, size, add = [], 0, len(line)
        buf.append(line)
        size += add
    if buf:
        raw.append("\n".join(buf))
    out, in_code = [], False
    for piece in raw:
        text = f"{FENCE}\n{piece}" if in_code else piece
        for line in piece.split("\n"):
            if line.lstrip().startswith(FENCE):
                in_code = not in_code
        if in_code:
            text += f"\n{FENCE}"
        if text.strip() and text.strip() != f"{FENCE}\n{FENCE}":
            out.append(text)
    return out


def _pack(blocks: list[str], limit: int) -> list[str]:
    """Pack blocks into pieces of at most `limit` characters (hard bound)."""
    if not isinstance(limit, int) or limit < MIN_CHUNK_CHARS:
        raise ValueError(f"chunk limit must be an integer >= {MIN_CHUNK_CHARS}, got {limit!r}")
    pieces, cur = [], ""
    for b in blocks:
        if len(b) > limit:  # oversized single block: split it, keeping code fences balanced
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_split_block(b, limit))
            continue
        if cur and len(cur) + len(b) + 2 > limit:
            pieces.append(cur)
            cur = ""
        cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        pieces.append(cur)
    return pieces


def chunk_page(html: str, url: str, limit: int = config.CHUNK_MAX_CHARS) -> list[Chunk]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one('div.body[role="main"]') or soup.select_one("div.body") or soup.body
    if body is None:
        return []
    for junk in body.select("a.headerlink, script, style, .toctree-wrapper"):
        junk.decompose()
    title = _clean(soup.title.get_text()).split(" — ")[0] if soup.title else url
    units: list[Unit] = []
    _units(body, [], "", units)
    chunks: list[Chunk] = []
    for crumbs, anchor, kind, blocks in units:
        crumbs = [c for c in crumbs if c]
        if not crumbs or crumbs[0] != title:
            crumbs = [title, *crumbs]
        breadcrumb = " > ".join(dict.fromkeys(crumbs))
        for piece in _pack(blocks, limit):
            if len(piece) < 40 and kind == "prose":
                continue  # drop near-empty fragments (e.g. a lone heading wrapper)
            cid = f"{url}#{anchor}::{len(chunks)}"
            chunks.append(Chunk(cid, url, anchor, breadcrumb, piece, kind))
    return chunks
