"""Client for the text generation model served on ROCm.

Structurally identical to mini-challenge 2's ``tabula_ocr.vlm.client`` —
retries with jittered backoff on transport/5xx errors only, a circuit
breaker, and ``guided_json`` pass-through for structured output — adapted for
text-only chat completions (no image content blocks) since this service
never sends images. The duplication between the two clients is deliberate:
each mini-challenge is an independently deployable service, and factoring a
shared HTTP client out into a third package would add a coordination cost
neither service actually needs at this scale.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from tabula_rag.config import Settings
from tabula_rag.errors import CircuitOpenError, LLMProtocolError, LLMTimeoutError
from tabula_rag.observability import LLM_FAILURES, LLM_LATENCY, get_logger

__all__ = ["CircuitBreaker", "LLMClient", "LLMResponse", "OpenAICompatibleLLM"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalised view of one text completion."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "unknown"
    latency_ms: float = 0.0

    def as_json(self) -> dict[str, Any]:
        """Parse the completion as a JSON object, tolerating a fenced code block."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text.removeprefix("json").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMProtocolError(
                "model response was not valid JSON",
                detail=f"{exc}; first 200 chars: {text[:200]!r}",
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMProtocolError(
                "model response was valid JSON but not an object",
                detail=f"got {type(parsed).__name__}",
            )
        return parsed


class LLMClient(Protocol):
    """Interface the pipeline depends on."""

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        guided_json: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        operation: str = "generate",
    ) -> LLMResponse:
        """Run one text completion."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...


@dataclass
class CircuitBreaker:
    """Minimal three-state breaker: closed, open, half-open."""

    failure_threshold: int
    reset_timeout_s: float
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        """True while the breaker is shedding load."""
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_timeout_s:
            self._opened_at = None
            self._failures = self.failure_threshold - 1
            return False
        return True

    def record_success(self) -> None:
        """Reset the failure count and close the breaker."""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Count a failure and open the breaker once the threshold is reached."""
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class OpenAICompatibleLLM:
    """Production client for a vLLM/SGLang text-generation endpoint."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Build a client; an injected transport is used by tests and benchmarks."""
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(
                settings.llm_timeout_s, connect=settings.llm_connect_timeout_s
            ),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            limits=httpx.Limits(max_connections=settings.llm_max_concurrency * 2),
        )
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
        self._breaker = CircuitBreaker(
            settings.breaker_failure_threshold, settings.breaker_reset_timeout_s
        )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        guided_json: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        operation: str = "generate",
    ) -> LLMResponse:
        """Run one completion, retrying transient failures."""
        if self._breaker.is_open:
            LLM_FAILURES.labels(reason="circuit_open").inc()
            raise CircuitOpenError(
                "language model circuit breaker is open",
                detail="upstream has failed repeatedly; shedding load",
            )

        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if guided_json is not None:
            payload["guided_json"] = guided_json
            payload["extra_body"] = {"guided_json": guided_json}

        last_error: Exception | None = None
        for attempt in range(self._settings.llm_max_retries + 1):
            started = time.perf_counter()
            try:
                async with self._semaphore:
                    response = await self._client.post("/chat/completions", json=payload)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"upstream {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    LLM_FAILURES.labels(reason=f"http_{response.status_code}").inc()
                    raise LLMProtocolError(
                        f"language model rejected the request ({response.status_code})",
                        detail=response.text[:500],
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                LLM_LATENCY.labels(operation=operation).observe(elapsed_ms / 1000)
                self._breaker.record_success()
                return self._parse(response.json(), elapsed_ms)
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                self._breaker.record_failure()
                reason = type(exc).__name__
                LLM_FAILURES.labels(reason=reason).inc()
                if attempt == self._settings.llm_max_retries:
                    break
                backoff = random.uniform(0, min(8.0, 0.5 * (2**attempt)))
                _log.warning(
                    "llm_call_retry",
                    attempt=attempt + 1,
                    backoff_s=round(backoff, 3),
                    reason=reason,
                    operation=operation,
                )
                await asyncio.sleep(backoff)

        raise LLMTimeoutError(
            "language model did not return a usable response",
            detail=f"{type(last_error).__name__}: {last_error}",
        )

    @staticmethod
    def _parse(body: dict[str, Any], elapsed_ms: float) -> LLMResponse:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProtocolError(
                "language model response was missing a completion", detail=str(body)[:500]
            ) from exc
        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            model=str(body.get("model", "unknown")),
            latency_ms=round(elapsed_ms, 2),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP transport."""
        await self._client.aclose()
