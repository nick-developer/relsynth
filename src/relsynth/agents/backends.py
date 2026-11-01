# SPDX-License-Identifier: Apache-2.0
"""Where an agent's judgement comes from.

``rules``      the default: deterministic judgements from the evidence, no
               language model, nothing leaves the process.  Every result can
               be reproduced from the evidence.
``openai``     any server that speaks the OpenAI chat-completions protocol
               with JSON output: a local model under Ollama, vLLM, llama.cpp
               or LM Studio (on-premises, the default base URL is Ollama's),
               or an organisation's gateway.  The model is asked only where
               the rules are unsure (an ASK question, a semantic match) and
               to word explanations; its answers pass the same guard.
``scripted``   a fixed function, for tests (including a model that follows
               every injected instruction, to test the guard).

A model never gets rows, never gets more than the task's data scope, and its
output is a JSON object that the guard checks before anyone sees it.  A
remote endpoint (not localhost) is refused unless ``allow_remote`` is set,
and then gets at most the ``aggregates`` scope (Prop. 1600).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .untrusted import SYSTEM_RULE


class Backend(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, schema: dict) -> dict: ...


@dataclass
class RulesBackend:
    """No model: the review, explain and doctor agents use their deterministic
    judgement only (``complete`` is never called)."""
    name: str = "rules"
    model: str = "relsynth-rules-1"
    uses_model: bool = False

    def complete(self, system: str, user: str, schema: dict) -> dict:  # pragma: no cover - never called
        raise RuntimeError("the rules backend has no language model")


@dataclass
class ScriptedBackend:
    """Answers from a function of (system, user, schema); for tests."""
    fn: Callable[[str, str, dict], Any]
    name: str = "scripted"
    model: str = "scripted"
    uses_model: bool = True
    calls: list = field(default_factory=list)

    def complete(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user})
        out = self.fn(system, user, schema)
        return out if isinstance(out, dict) else json.loads(out)


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]", "host.docker.internal", "ollama")


@dataclass
class OpenAICompatBackend:
    """A chat-completions endpoint with JSON-schema output."""
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b-instruct"
    api_key_env: str | None = None
    timeout_s: float = 120.0
    allow_remote: bool = False
    temperature: float = 0.0
    seed: int = 7
    name: str = "openai"
    uses_model: bool = True
    calls: int = 0
    seconds: float = 0.0

    def __post_init__(self):
        u = urllib.parse.urlparse(self.base_url)
        if u.scheme not in ("http", "https"):
            raise ValueError("the model endpoint must be an http(s) URL")
        if not self.local and not self.allow_remote:
            raise ValueError(f"the model endpoint {u.hostname} is not on this machine; set allow_remote: true to send "
                             "aggregates (never rows or values) to it")
        if not self.local and u.scheme != "https":
            raise ValueError("a remote model endpoint must use https")

    @property
    def local(self) -> bool:
        return (urllib.parse.urlparse(self.base_url).hostname or "") in LOCAL_HOSTS

    def complete(self, system: str, user: str, schema: dict) -> dict:
        body = {"model": self.model, "temperature": self.temperature, "seed": self.seed,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "answer", "schema": schema, "strict": True}}}
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",  # noqa: S310
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # noqa: S310  # nosec B310 (scheme checked)
            doc = json.loads(r.read())
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        text = doc["choices"][0]["message"]["content"] or "{}"
        try:
            return json.loads(text)
        except ValueError:
            # some servers wrap JSON in prose or a code fence: take the outermost object
            i, j = text.find("{"), text.rfind("}")
            return json.loads(text[i:j + 1]) if 0 <= i < j else {}


def make_backend(cfg: dict | str | None) -> Any:
    """A backend from a configuration: ``"rules"``, or a mapping such as
    ``{"kind": "openai", "base_url": ..., "model": ..., "api_key_env": ...}``.
    The environment can set it too: RELSYNTH_AGENT_BACKEND, RELSYNTH_AGENT_BASE_URL,
    RELSYNTH_AGENT_MODEL, RELSYNTH_AGENT_API_KEY_ENV."""
    if cfg is None:
        kind = os.environ.get("RELSYNTH_AGENT_BACKEND", "rules")
        cfg = {"kind": kind}
        for k, e in (("base_url", "RELSYNTH_AGENT_BASE_URL"), ("model", "RELSYNTH_AGENT_MODEL"),
                     ("api_key_env", "RELSYNTH_AGENT_API_KEY_ENV")):
            if os.environ.get(e):
                cfg[k] = os.environ[e]
    if isinstance(cfg, str):
        cfg = {"kind": cfg}
    cfg = dict(cfg)
    kind = cfg.pop("kind", "rules")
    if kind == "rules":
        return RulesBackend()
    if kind in ("openai", "ollama", "vllm", "local"):
        return OpenAICompatBackend(**{k: v for k, v in cfg.items()
                                      if k in ("base_url", "model", "api_key_env", "timeout_s", "allow_remote",
                                               "temperature", "seed")})
    raise ValueError(f"unknown agent backend {kind!r}; use rules or openai")


__all__ = ["Backend", "RulesBackend", "ScriptedBackend", "OpenAICompatBackend", "make_backend", "SYSTEM_RULE"]
