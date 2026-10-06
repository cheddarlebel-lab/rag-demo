from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import requests

from rag_demo import config
from rag_demo.llm import ClaudeLLM, LLMOutputError, OllamaLLM, get_llm
from tests.conftest import FakeResponse


class FakeMessages:
    def __init__(self, resp: Any):
        self.resp = resp
        self.kwargs: dict[str, Any] = {}

    def create(self, **kw: Any) -> Any:
        self.kwargs = kw
        return self.resp


def claude(resp: Any) -> tuple[ClaudeLLM, FakeMessages]:
    m = FakeMessages(resp)
    return ClaudeLLM(model="test-model", client=SimpleNamespace(messages=m)), m


def msg(stop: str, text: str | None = '{"a": 1}', **extra: Any) -> Any:
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(stop_reason=stop, content=content, **extra)


def test_claude_requires_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CLAUDE_MODEL", None)
    with pytest.raises(ValueError, match="RAG_CLAUDE_MODEL"):
        ClaudeLLM(client=object())


def test_claude_request_shape_and_success() -> None:
    llm, m = claude(msg("end_turn"))
    assert llm.complete_json("sys", "user", {"type": "object"}) == {"a": 1}
    assert m.kwargs["model"] == "test-model"
    assert m.kwargs["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert llm.name == "anthropic/test-model"


@pytest.mark.parametrize("resp,match", [
    (msg("refusal", None, stop_details=SimpleNamespace(category="cyber")), "declined"),
    (msg("max_tokens", '{"a": '), "truncated"),
    (msg("end_turn", None), "no text"),
    (msg("end_turn", "not json"), "invalid JSON"),
    (msg("end_turn", "[1, 2]"), "expected an object"),
])
def test_claude_failure_modes(resp: Any, match: str) -> None:
    llm, _ = claude(resp)
    with pytest.raises(LLMOutputError, match=match):
        llm.complete_json("s", "u", {})


def test_ollama_truncation_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies = iter([{"done_reason": "length", "message": {"content": '{"a":'}},
                   {"done_reason": "stop", "message": {"content": '{"a": 2}'}}])
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, next(bodies)))
    llm = OllamaLLM(model="m")
    with pytest.raises(LLMOutputError, match="truncated"):
        llm.complete_json("s", "u", {})
    assert llm.complete_json("s", "u", {}) == {"a": 2}


def test_unknown_provider() -> None:
    with pytest.raises(ValueError):
        get_llm("gpt")
