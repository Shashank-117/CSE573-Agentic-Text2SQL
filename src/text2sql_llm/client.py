"""Shared LLM interface for every LLM-based system (single-agent, two-agent).

Backends implement ``LLMClient.generate(messages, params) -> Generation``.
Token counts come from the backend (server ``usage`` field), never from
guesses, so cost tables are comparable across systems.

``OpenAICompatibleClient`` talks to any server exposing
``POST /v1/chat/completions`` (llama.cpp ``llama-server``, Ollama, vLLM,
LM Studio), so the same code runs on a laptop GPU or a teammate's box.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    seed: int | None = 0
    stop: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    finish_reason: str | None = None


class LLMError(RuntimeError):
    """Raised when the backend fails (network, HTTP error, bad response)."""


class LLMClient(Protocol):
    model: str

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> Generation:
        ...

    def describe(self) -> dict[str, Any]:
        ...


class OpenAICompatibleClient:
    """Minimal client for an OpenAI-compatible chat completions endpoint."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 300.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._api_key = api_key  # local servers ignore it; never logged

    def describe(self) -> dict[str, Any]:
        return {"backend": "openai_compatible", "base_url": self.base_url, "model": self.model}

    def _payload(self, messages: Sequence[Message], params: GenerationParams) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_new_tokens,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.stop:
            payload["stop"] = list(params.stop)
        return payload

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> Generation:
        body = json.dumps(self._payload(messages, params)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)

        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"request to {self.base_url} failed: {exc}") from exc
        latency = time.monotonic() - started
        return _parse_response(data, self.model, latency)


def _parse_response(data: dict[str, Any], model: str, latency: float) -> Generation:
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data["usage"]
        return Generation(
            text=text,
            input_tokens=int(usage["prompt_tokens"]),
            output_tokens=int(usage["completion_tokens"]),
            latency_s=latency,
            model=str(data.get("model") or model),
            finish_reason=choice.get("finish_reason"),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"unexpected response shape: {str(data)[:300]}") from exc


@dataclass
class ScriptedLLM:
    """Test double: returns scripted responses in order and records prompts."""

    responses: list[str]
    model: str = "scripted"
    calls: list[list[Message]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {"backend": "scripted", "model": self.model}

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> Generation:
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        self.calls.append(list(messages))
        text = self.responses.pop(0)
        prompt_chars = sum(len(m.content) for m in messages)
        return Generation(text=text, input_tokens=max(1, prompt_chars // 4),
                          output_tokens=max(1, len(text) // 4), latency_s=0.0, model=self.model)
