"""Runtime configuration, sourced once from the environment.

Mirrors the pattern from mini-challenge 2's ``tabula_ocr.config``: every knob
lives here, nothing reads ``os.environ`` elsewhere, and the whole surface is
greppable in one file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Service configuration, sourced from ``TABULA_RAG_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TABULA_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "tabula-rag"
    environment: Literal["dev", "staging", "prod", "test"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- Upstream text LLM (vLLM on ROCm, OpenAI-compatible) --------------
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-14B-Instruct"
    llm_api_key: str = Field(default="EMPTY", repr=False)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_connect_timeout_s: float = Field(default=5.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_max_concurrency: int = Field(default=8, ge=1, le=256)

    # --- Embeddings (a second small model served on the same MI300X) -----
    embedding_base_url: str = "http://localhost:8001/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, ge=8)

    # --- Reranker ----------------------------------------------------------
    reranker_base_url: str = "http://localhost:8002/v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Circuit breaker ----------------------------------------------------
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_timeout_s: float = Field(default=30.0, gt=0)

    # --- Chunking ------------------------------------------------------------
    chunk_target_tokens: int = Field(default=180, ge=20, le=2000)
    chunk_overlap_tokens: int = Field(default=30, ge=0, le=500)

    # --- Retrieval -----------------------------------------------------------
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    rerank_top_k: int = Field(default=4, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, description="RRF's rank-damping constant.")
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)

    # --- Verification & abstention --------------------------------------------
    claim_coverage_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Below this lexical-coverage score, a claim is UNSUPPORTED.",
    )
    max_unsupported_ratio: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Above this share of unsupported claims, the whole answer is withheld.",
    )
    min_retrieval_score: float = Field(
        default=0.02,
        ge=0.0,
        description="Below this fused score, retrieval is treated as empty (abstain).",
    )

    # --- Cost accounting -------------------------------------------------------
    gpu_hourly_rate_usd: float = Field(default=1.99, ge=0)
    measured_tokens_per_s: float = Field(default=900.0, gt=0)

    @field_validator("llm_base_url", "embedding_base_url", "reranker_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """Normalise base URLs so path joins never double up."""
        return value.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
