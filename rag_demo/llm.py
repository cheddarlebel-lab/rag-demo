"""Pluggable LLM interface.

Any backend implements one method: `complete_json(system, user, schema) -> dict`.
The analysis layer never knows which model it is talking to.

* OllamaLLM  (default) - local, free. Uses ollama's JSON-schema constrained output.
* ClaudeLLM  (optional) - Anthropic Messages API via the official `anthropic` SDK,
  using structured outputs (output_config.format). NOT called by this repo's demo,
  eval or site build; enable with RAG_LLM=claude + ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

import requests

from . import config


class LLM(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


class OllamaLLM:
    def __init__(self, model: str = config.CHAT_MODEL, host: str = config.OLLAMA_HOST):
        self.model, self.host = model, host
        self.name = f"ollama/{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        resp = requests.post(f"{self.host}/api/chat", timeout=600, json={
            "model": self.model,
            "stream": False,
            "format": schema,                       # constrained decoding to the schema
            "options": {"temperature": 0, "num_ctx": 8192, "seed": 7},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        })
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])


class ClaudeLLM:
    """Optional adapter. Requires `pip install anthropic` and ANTHROPIC_API_KEY.

    Costs money per call - which is why nothing in this repo calls it by default.
    """

    def __init__(self, model: str = os.environ.get("RAG_CLAUDE_MODEL", "claude-opus-5-5")):
        import anthropic  # imported lazily so the default stack has no dependency on it
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic/{model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


def get_llm(provider: str = config.LLM_PROVIDER) -> LLM:
    if provider == "ollama":
        return OllamaLLM()
    if provider == "claude":
        return ClaudeLLM()
    raise ValueError(f"unknown LLM provider: {provider!r} (expected 'ollama' or 'claude')")
