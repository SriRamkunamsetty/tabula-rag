"""A deterministic, GPU-free stand-in for the text generation model.

Mirrors ``tabula_ocr.vlm.fake.ScriptedVLM``: a scripted sequence of responses
used by the unit tests and by ``tabula demo``. Can be told to produce a
well-cited answer, an uncited answer, or an answer that cites a real chunk
but says something that chunk doesn't support — which is exactly the
scenario ``verification.py`` exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tabula_rag.llm.client import LLMResponse

__all__ = ["ScriptedLLM"]


@dataclass
class ScriptedLLM:
    """Replays a fixed list of response strings, one per call, in order."""

    responses: list[str]
    model: str = "scripted-llm"
    latency_ms: float = 8.0
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)

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
        """Return the next scripted response."""
        self.calls.append(
            {
                "operation": operation,
                "temperature": temperature,
                "guided": guided_json is not None,
                "system": system,
                "user": user,
            }
        )
        text = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return LLMResponse(
            text=text,
            prompt_tokens=200,
            completion_tokens=max(1, len(text) // 4),
            model=self.model,
            latency_ms=self.latency_ms,
        )

    async def aclose(self) -> None:
        """No-op; present so the fake satisfies the client protocol."""
        return None
