"""Structured logging.

`event(logger, "crawl.fetched", url=..., status=200)` emits one record with key/value
fields. Output goes to stderr (stdout stays clean for JSON results):

* RAG_LOG_FORMAT=text (default): `12:00:01 INFO crawl.fetched url=... status=200`
* RAG_LOG_FORMAT=json: one JSON object per line, for log shippers.

RAG_LOG_LEVEL sets the level (default INFO).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_CONFIGURED = False


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields: dict[str, Any] = getattr(record, "fields", {})
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        ts = self.formatTime(record, "%H:%M:%S")
        return f"{ts} {record.levelname} {record.getMessage()}" + (f" {kv}" if kv else "")


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                               "level": record.levelname, "logger": record.name,
                               "event": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        return json.dumps(out, default=str)


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever sys.stderr is at emit time (test runners swap it)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def configure(level: str | None = None, fmt: str | None = None) -> None:
    global _CONFIGURED
    root = logging.getLogger("rag_demo")
    handler = _StderrHandler(sys.stderr)
    fmt = fmt or os.environ.get("RAG_LOG_FORMAT", "text")
    handler.setFormatter(_JSONFormatter() if fmt == "json" else _TextFormatter())
    root.handlers[:] = [handler]
    root.setLevel(level or os.environ.get("RAG_LOG_LEVEL", "INFO"))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure()
    return logging.getLogger(name)


def event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})
