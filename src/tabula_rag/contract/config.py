"""Contract-mode settings, read straight from the environment.

Deliberately not ``pydantic-settings``: the graded harness starts a new process for every
question, and this mode should spend its 30 seconds on the question, not on importing
a settings framework. Every knob is an ``TABULA_RAG_*`` variable with a safe default.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ContractSettings"]


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _default_index_dir() -> Path:
    """``/app/index`` when writable (the container), otherwise a temp directory."""
    app = Path("/app")
    if app.is_dir() and os.access(app, os.W_OK):
        return app / "index"
    return Path(tempfile.gettempdir()) / "tabula-rag-index"


@dataclass(frozen=True)
class ContractSettings:
    """All contract-mode knobs."""

    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = ""  # empty: use whichever model the server reports first
    llm_api_key: str = "EMPTY"
    index_dir: Path = field(default_factory=_default_index_dir)
    output_dir: Path = Path("/app/output")

    query_budget_s: float = 24.0  # hard ceiling is 30 s per question, process start included
    llm_timeout_s: float = 18.0
    index_llm_wait_s: float = 420.0  # how long --index waits for the model server to come up
    vision_timeout_s: float = 90.0
    vision_concurrency: int = 4
    image_max_side: int = 1600
    top_k: int = 10

    @classmethod
    def from_env(cls) -> ContractSettings:
        """Build settings from ``TABULA_RAG_*`` environment variables."""
        env = os.environ
        default = cls()
        return cls(
            llm_base_url=env.get("TABULA_RAG_LLM_BASE_URL", default.llm_base_url).rstrip("/"),
            llm_model=env.get("TABULA_RAG_LLM_MODEL", default.llm_model),
            llm_api_key=env.get("TABULA_RAG_LLM_API_KEY", default.llm_api_key),
            index_dir=Path(env.get("TABULA_RAG_INDEX_DIR", str(default.index_dir))),
            output_dir=Path(env.get("TABULA_RAG_OUTPUT_DIR", str(default.output_dir))),
            query_budget_s=_float("TABULA_RAG_QUERY_BUDGET_S", default.query_budget_s),
            llm_timeout_s=_float("TABULA_RAG_CONTRACT_LLM_TIMEOUT_S", default.llm_timeout_s),
            index_llm_wait_s=_float("TABULA_RAG_INDEX_LLM_WAIT_S", default.index_llm_wait_s),
            vision_timeout_s=_float("TABULA_RAG_VISION_TIMEOUT_S", default.vision_timeout_s),
            vision_concurrency=_int(
                "TABULA_RAG_VISION_CONCURRENCY", default.vision_concurrency
            ),
            image_max_side=_int("TABULA_RAG_IMAGE_MAX_SIDE", default.image_max_side),
            top_k=_int("TABULA_RAG_CONTRACT_TOP_K", default.top_k),
        )
