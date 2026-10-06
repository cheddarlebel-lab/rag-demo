"""Pluggable LLM interface.

Any backend implements one method: `complete_json(system, user, schema) -> dict`.
The analysis layer never knows which model it is talking to, and it validates
whatever comes back against the schema itself (see pipeline.validate_output).

* OllamaLLM  (default) - local, free. Uses ollama's JSON-schema constrained output.
* ClaudeLLM  (optional) - Anthropic Messages API via the official `anthropic` SDK,
  using structured outputs (output_config.format). Implemented and covered by
  mocked unit tests; NOT exercised against the live API by this repo (no paid calls).
  Enable with RAG_LLM=claude, RAG_CLAUDE_MODEL=<model id> and ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

import requests

from . import config


class LLMOutputError(RuntimeError):
    """The model returned something unusable: truncated, refused, empty or not JSON."""


class LLM(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


def _parse_json(text: str, who: str) -> dict[str, Any]:
    try:
        out = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"{who} returned invalid JSON: {exc}") from exc
    if not isinstance(out, dict):
        raise LLMOutputError(f"{who} returned JSON {type(out).__name__}, expected an object")
    return out


class OllamaLLM:
    def __init__(self, model: str = config.CHAT_MODEL, host: str = config.OLLAMA_HOST):
        self.model, self.host = model, host
        self.name = f"ollama/{model}"

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(f"{self.host}/api/chat", timeout=600, json={
            "model": self.model,
            "stream": False,
            "format": schema,                       # constrained decoding to the schema
            "options": {"temperature": 0, "num_ctx": 8192, "seed": 7},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        })
        resp.raise_for_status()
        body = resp.json()
        if body.get("done_reason") == "length":
            raise LLMOutputError(f"{self.name} hit its output limit; the JSON is truncated")
        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            raise LLMOutputError(f"{self.name} returned an empty response")
        return _parse_json(content, self.name)


class ClaudeLLM:
    """Optional adapter. Requires `pip install anthropic`, ANTHROPIC_API_KEY and RAG_CLAUDE_MODEL.

    Costs money per call - which is why nothing in this repo calls it by default.
    """

    def __init__(self, model: str | None = None, client: Any = None):
        model = model or config.CLAUDE_MODEL
        if not model:
            raise ValueError(
                "RAG_LLM=claude needs an explicit model: set RAG_CLAUDE_MODEL to a model id "
                "your Anthropic account can use (for example claude-opus-5-5).")
        if client is None:
            try:
                import anthropic  # imported lazily so the default stack has no dependency on it
            except ImportError as exc:
                raise ValueError("RAG_LLM=claude needs the SDK: pip install -r requirements-claude.txt") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.name = f"anthropic/{model}"

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMOutputError(f"{self.name} declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMOutputError(f"{self.name} hit max_tokens; the JSON is truncated")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if not text:
            raise LLMOutputError(f"{self.name} returned no text block (stop_reason={response.stop_reason})")
        return _parse_json(text, self.name)


def get_llm(provider: str = config.LLM_PROVIDER) -> LLM:
    if provider == "ollama":
        return OllamaLLM()
    if provider == "claude":
        return ClaudeLLM()
    raise ValueError(f"unknown LLM provider: {provider!r} (expected 'ollama' or 'claude')")
