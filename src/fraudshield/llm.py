"""Thin LLM client: Claude when an API key is configured, deterministic offline fallback otherwise.

The offline path keeps the project runnable in CI and for reviewers without API keys:
every call site passes a `fallback` callable that produces the same Pydantic type
from heuristics, so behaviour degrades gracefully instead of crashing.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")


@dataclass
class CallStats:
    calls: int = 0
    fallbacks: int = 0
    latency_ms: list[float] = field(default_factory=list)


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, offline: bool | None = None):
        self.model = model
        self.stats = CallStats()
        if offline is None:
            offline = os.getenv("LLM_OFFLINE", "").lower() in {"1", "true"} or not os.getenv(
                "ANTHROPIC_API_KEY"
            )
        self.offline = offline
        self._client = None
        if not self.offline:
            import anthropic  # imported lazily so offline mode has no hard dependency

            self._client = anthropic.Anthropic()

    @property
    def mode(self) -> str:
        return "offline-heuristic" if self.offline else f"anthropic:{self.model}"

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
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": _strict_schema(schema)},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if response.stop_reason == "refusal":
                raise RuntimeError("model declined the request")
            text = next(b.text for b in response.content if b.type == "text")
            return schema.model_validate(json.loads(text))
        except Exception as exc:  # network, refusal, schema drift -> degrade, never crash
            log.warning("LLM call failed (%s); using heuristic fallback", exc)
            self.stats.fallbacks += 1
            return fallback()
        finally:
            self.stats.latency_ms.append((time.perf_counter() - start) * 1000)


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
