"""Thin LLM client: Groq or Claude when a key is configured, deterministic offline fallback otherwise.

Provider order (first match wins):
  1. LLM_OFFLINE=1                -> offline heuristics (CI, reproducible evals)
  2. LLM_PROVIDER=groq|anthropic  -> that provider, if its key is set
  3. GROQ_API_KEY                 -> Groq (free tier, OpenAI-compatible API; used for demos)
  4. ANTHROPIC_API_KEY            -> Claude
  5. otherwise                    -> offline heuristics

Every call site passes a `fallback` callable that produces the same Pydantic type from heuristics, so a
network error, rate limit or schema drift degrades to the offline answer instead of crashing.
Keys are read from the environment or a `.env` file in the working directory or a parent directory
(never commit it).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"


def load_dotenv(start: Path | None = None) -> None:
    """Load KEY=VALUE lines from the nearest .env (cwd upwards) without overriding real env vars."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        f = d / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            return


def pick_provider() -> str:
    """'offline', 'groq' or 'anthropic' from the environment (see module docstring)."""
    load_dotenv()
    if os.getenv("LLM_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return "offline"
    keys = {"groq": os.getenv("GROQ_API_KEY"), "anthropic": os.getenv("ANTHROPIC_API_KEY")}
    wanted = os.getenv("LLM_PROVIDER", "").lower()
    if wanted in keys:
        return wanted if keys[wanted] else "offline"
    return next((p for p in ("groq", "anthropic") if keys[p]), "offline")


@dataclass
class CallStats:
    calls: int = 0
    fallbacks: int = 0
    latency_ms: list[float] = field(default_factory=list)


class LLMClient:
    def __init__(self, model: str | None = None, offline: bool | None = None, provider: str | None = None):
        self.provider = "offline" if offline else (provider or pick_provider())
        self.offline = self.provider == "offline"
        default = GROQ_DEFAULT_MODEL if self.provider == "groq" else ANTHROPIC_DEFAULT_MODEL
        wanted = model or os.getenv("LLM_MODEL") or default
        # a Claude model id left in LLM_MODEL (or a config default) must not be sent to Groq, and vice versa
        self.model = wanted if wanted.startswith("claude") == (self.provider != "groq") else default
        self.stats = CallStats()
        self._client = None
        if self.provider == "anthropic":
            import anthropic  # imported lazily so offline/Groq modes have no hard dependency

            self._client = anthropic.Anthropic()

    @property
    def mode(self) -> str:
        return "offline-heuristic" if self.offline else f"{self.provider}:{self.model}"

    def structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        fallback: Callable[[], T],
        effort: str = "medium",
    ) -> T:
        """Return a validated `schema` instance from the model, or from `fallback`."""
        self.stats.calls += 1
        if self.offline:
            self.stats.fallbacks += 1
            return fallback()
        start = time.perf_counter()
        try:
            if self.provider == "groq":
                text = self._groq(system, prompt, schema, effort)
            else:
                text = self._anthropic(system, prompt, schema, effort)
            return schema.model_validate(json.loads(text))
        except Exception as exc:  # network, refusal, rate limit, schema drift -> degrade, never crash
            log.warning("LLM call failed (%s); using heuristic fallback", exc)
            self.stats.fallbacks += 1
            return fallback()
        finally:
            self.stats.latency_ms.append((time.perf_counter() - start) * 1000)

    # -- providers ---------------------------------------------------------------------------------
    def _anthropic(self, system: str, prompt: str, schema: type[BaseModel], effort: str) -> str:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": _strict_schema(schema)}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request")
        return next(b.text for b in response.content if b.type == "text")

    def _groq(self, system: str, prompt: str, schema: type[BaseModel], effort: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_completion_tokens": 8192,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": _strict_schema(schema)},
            },
        }
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = effort if effort in {"low", "medium", "high"} else "medium"
        try:
            data = groq_chat(body)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            # model without strict json_schema support: ask for plain JSON and let Pydantic validate
            body["response_format"] = {"type": "json_object"}
            body["messages"][0]["content"] += ("\n\nReply with one JSON object matching this JSON schema:\n"
                                               + json.dumps(schema.model_json_schema()))
            data = groq_chat(body)
        return data["choices"][0]["message"]["content"]


def groq_chat(body: dict, retries: int = 3, timeout: float = 60.0) -> dict:
    """POST a chat completion to Groq; waits and retries on 429 / 5xx (free-tier rate limits)."""
    req = urllib.request.Request(
        GROQ_URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json",
                 "User-Agent": "portfolio-llm-client/1.0"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if attempt == retries or exc.code not in (429, 500, 502, 503):
                raise
            wait = float(exc.headers.get("retry-after") or 2 ** attempt)
            log.info("Groq %s; retrying in %.1fs", exc.code, wait)
            time.sleep(min(wait, 30.0))
    raise RuntimeError("unreachable")


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with additionalProperties=false on every object (required by strict output)."""
    schema = model.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                node["required"] = list(node.get("properties", {}).keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema
