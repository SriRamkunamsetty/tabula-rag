"""A small OpenAI-compatible client for contract mode: JSON answers and image reading.

Every method degrades to ``None`` instead of raising: in this mode a model that is slow,
down, or confused must produce an *empty answer*, never a crash and never a missing output
file. Callers treat ``None`` as "no usable model output".
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from typing import Any, Protocol

import httpx

from tabula_rag.contract.config import ContractSettings
from tabula_rag.observability import get_logger

__all__ = ["ContractLLM", "ModelClient", "OpenAIModelClient", "extract_json_object"]

_log = get_logger(__name__)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

TRANSCRIBE_PROMPT = (
    "Transcribe every piece of text visible in this image, exactly as written, preserving "
    "case, punctuation and identifiers (part numbers, revisions, codes, pin names). "
    "If the image is a diagram, table, label or pinout, list each labelled item on its own "
    "line as 'label: value' (for example 'Pin B14: THERM_ALERT#'). "
    "Output plain text only, with no commentary."
)


class ModelClient(Protocol):
    """What the answering and indexing code needs from a model."""

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        """Return a JSON object answer, or ``None`` if the model gave nothing usable."""
        ...

    async def transcribe(self, image_bytes: bytes, *, timeout_s: float) -> str | None:
        """Return the text visible in an image, or ``None``."""
        ...

    async def wait_ready(self, budget_s: float) -> bool:
        """Block until the model server answers, or ``budget_s`` runs out."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of model text (tolerates fences and chatter)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def prepare_image(image_bytes: bytes, max_side: int) -> tuple[bytes, str]:
    """Decode any supported image, flatten to RGB, cap its size, and re-encode as PNG."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        rgb = image.convert("RGB")
        longest = max(rgb.size)
        if longest > max_side:
            scale = max_side / longest
            rgb = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            )
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


class OpenAIModelClient:
    """Talks to a vLLM/SGLang endpoint over the OpenAI chat-completions API."""

    def __init__(
        self, settings: ContractSettings, client: httpx.AsyncClient | None = None
    ) -> None:
        """Create a client; tests inject a transport-backed ``httpx.AsyncClient``."""
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )
        self._model = settings.llm_model or None
        self._schema_supported = True

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _resolve_model(self, timeout_s: float) -> str | None:
        if self._model:
            return self._model
        try:
            response = await self._client.get("/models", timeout=timeout_s)
            response.raise_for_status()
            data = response.json().get("data") or []
            self._model = data[0]["id"] if data else None
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            self._model = None
        return self._model

    async def wait_ready(self, budget_s: float) -> bool:
        """Poll ``/models`` until it answers with a model, up to ``budget_s`` seconds."""
        deadline = time.monotonic() + max(0.0, budget_s)
        while True:
            if await self._resolve_model(timeout_s=3.0):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(2.0)

    async def _complete(self, payload: dict[str, Any], timeout_s: float) -> str | None:
        try:
            response = await self._client.post(
                "/chat/completions", json=payload, timeout=timeout_s
            )
        except httpx.HTTPError as exc:
            _log.warning("llm_transport_error", error=type(exc).__name__)
            return None
        if response.status_code == 400 and "response_format" in payload:
            self._schema_supported = False  # server rejected structured output; retry plain
            payload = {k: v for k, v in payload.items() if k != "response_format"}
            return await self._complete(payload, timeout_s)
        if response.status_code >= 400:
            _log.warning("llm_http_error", status=response.status_code)
            return None
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError):
            return None
        return str(content) if content is not None else None

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        """Ask for a JSON object matching ``schema``; ``None`` if nothing usable came back."""
        model = await self._resolve_model(min(timeout_s, 5.0))
        if not model:
            return None
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if self._schema_supported:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            }
        text = await self._complete(payload, timeout_s)
        return extract_json_object(text) if text else None

    async def transcribe(self, image_bytes: bytes, *, timeout_s: float) -> str | None:
        """Read the text in an image with the vision-language model."""
        model = await self._resolve_model(min(timeout_s, 5.0))
        if not model:
            return None
        try:
            png, mime = prepare_image(image_bytes, self._settings.image_max_side)
        except Exception as exc:
            _log.warning("image_decode_failed", error=type(exc).__name__)
            return None
        data_uri = f"data:{mime};base64,{base64.b64encode(png).decode('ascii')}"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
        }
        text = await self._complete(payload, timeout_s)
        return text.strip() if text and text.strip() else None


ContractLLM = OpenAIModelClient
