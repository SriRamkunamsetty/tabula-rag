"""Tests for the upstream text LLM client.

Uses ``httpx.MockTransport`` to assert retry, backoff, and circuit-breaker
behaviour exactly — mirrors ``tabula_ocr``'s equivalent client tests from
mini-challenge 2.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tabula_rag.config import Settings
from tabula_rag.errors import CircuitOpenError, LLMProtocolError, LLMTimeoutError
from tabula_rag.llm.client import OpenAICompatibleLLM


def completion(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "model": "test-llm",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def build_client(handler, **overrides) -> OpenAICompatibleLLM:
    settings = Settings(
        environment="test",
        llm_max_retries=overrides.pop("llm_max_retries", 2),
        breaker_failure_threshold=overrides.pop("breaker_failure_threshold", 5),
        breaker_reset_timeout_s=overrides.pop("breaker_reset_timeout_s", 60.0),
        log_level="ERROR",
        **overrides,
    )
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url=settings.llm_base_url)
    return OpenAICompatibleLLM(settings, client=http)


async def call(client: OpenAICompatibleLLM, **kwargs):
    defaults = {"system": "s", "user": "u", "temperature": 0.0}
    defaults.update(kwargs)
    return await client.complete(**defaults)


class TestPayloadShape:
    @pytest.mark.asyncio
    async def test_guided_json_is_sent_both_ways(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=completion('{"ok": 1}'))

        client = build_client(handler)
        await call(client, guided_json={"type": "object"})
        assert captured["guided_json"] == {"type": "object"}
        assert captured["extra_body"]["guided_json"] == {"type": "object"}
        await client.aclose()

    @pytest.mark.asyncio
    async def test_usage_is_propagated(self):
        client = build_client(
            lambda r: httpx.Response(
                200, json=completion("{}", prompt_tokens=700, completion_tokens=120)
            )
        )
        response = await call(client)
        assert response.prompt_tokens == 700
        assert response.completion_tokens == 120
        await client.aclose()


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_server_errors_are_retried_then_surface_as_timeout(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503, text="overloaded")

        client = build_client(handler, llm_max_retries=2)
        with pytest.raises(LLMTimeoutError):
            await call(client)
        assert attempts["n"] == 3
        await client.aclose()

    @pytest.mark.asyncio
    async def test_client_errors_are_not_retried(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(400, text="bad request")

        client = build_client(handler, llm_max_retries=3)
        with pytest.raises(LLMProtocolError):
            await call(client)
        assert attempts["n"] == 1
        await client.aclose()

    @pytest.mark.asyncio
    async def test_transport_error_recovers_on_retry(self):
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json=completion('{"ok": true}'))

        client = build_client(handler, llm_max_retries=2)
        response = await call(client)
        assert response.as_json() == {"ok": True}
        await client.aclose()

    @pytest.mark.asyncio
    async def test_breaker_opens_and_sheds_load(self):
        client = build_client(
            lambda r: httpx.Response(500, text="boom"),
            llm_max_retries=0,
            breaker_failure_threshold=2,
            breaker_reset_timeout_s=60.0,
        )
        for _ in range(2):
            with pytest.raises(LLMTimeoutError):
                await call(client)
        with pytest.raises(CircuitOpenError):
            await call(client)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_missing_completion_is_a_protocol_error(self):
        client = build_client(lambda r: httpx.Response(200, json={"choices": []}))
        with pytest.raises(LLMProtocolError):
            await call(client)
        await client.aclose()
