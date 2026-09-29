"""HTTP surface.

Same design as mini-challenge 2's API: liveness and readiness are separate
probes, every error maps through one typed hierarchy to a stable JSON body,
and nothing here holds business logic — it all lives in
:class:`RAGService`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from tabula_rag import __version__
from tabula_rag.config import Settings, get_settings
from tabula_rag.corpus_index import CorpusIndex
from tabula_rag.errors import TabulaRagError
from tabula_rag.llm.client import OpenAICompatibleLLM
from tabula_rag.models import AnswerResult, IngestRequest, IngestResult, QueryRequest
from tabula_rag.observability import (
    INFLIGHT,
    QUERIES,
    QUERY_LATENCY,
    bind_request_id,
    configure_logging,
    get_logger,
    new_request_id,
)
from tabula_rag.pipeline import RAGService
from tabula_rag.prompts.registry import DEFAULT_VERSION, list_versions
from tabula_rag.retrieval.embeddings import OpenAICompatibleEmbeddings
from tabula_rag.retrieval.reranker import OpenAICompatibleReranker

__all__ = ["create_app", "router"]

_log = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness payload, including which dependencies were checked."""

    ready: bool
    checks: dict[str, str]


class InfoResponse(BaseModel):
    """Discovery payload."""

    documents_indexed: int
    chunks_indexed: int
    prompt_versions: list[dict[str, str]]
    default_prompt_version: str


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    """Build the ASGI application."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.service = service
        else:
            llm = OpenAICompatibleLLM(resolved)
            embeddings = OpenAICompatibleEmbeddings(resolved)
            reranker = OpenAICompatibleReranker(resolved)
            app.state.service = RAGService(
                resolved, llm, embeddings, reranker, CorpusIndex(resolved, embeddings)
            )
        app.state.settings = resolved
        _log.info(
            "service_started",
            version=__version__,
            environment=resolved.environment,
            model=resolved.llm_model,
        )
        try:
            yield
        finally:
            await app.state.service._llm.aclose()
            _log.info("service_stopped")

    app = FastAPI(
        title="TABULA RAG",
        version=__version__,
        summary=(
            "Grounded question answering over proprietary documents, "
            "with claim-level verification."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def correlate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or new_request_id()
        bind_request_id(request_id)
        INFLIGHT.inc()
        try:
            response: Response = await call_next(request)
        finally:
            INFLIGHT.dec()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(TabulaRagError)
    async def handle_tabula_error(_request: Request, exc: TabulaRagError) -> JSONResponse:
        _log.warning("request_failed", code=exc.code, message=exc.message)
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    return app


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness. Deliberately checks nothing external."""
    return HealthResponse(status="ok", service="tabula-rag", version=__version__)


@router.get("/readyz", response_model=ReadyResponse, tags=["ops"])
async def readyz(request: Request) -> ReadyResponse:
    """Readiness. Verifies the corpus is non-empty and the LLM responds."""
    checks: dict[str, str] = {}
    service: RAGService = request.app.state.service

    checks["corpus"] = f"ok ({len(service.corpus)} chunks)" if len(service.corpus) else "empty"
    try:
        await service._llm.complete(
            system="ping", user="ping", max_tokens=1, operation="readiness"
        )
        checks["llm"] = "ok"
    except Exception as exc:
        checks["llm"] = f"error: {type(exc).__name__}"

    return ReadyResponse(ready=all(v.startswith("ok") for v in checks.values()), checks=checks)


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition."""
    body: bytes = generate_latest()
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/info", response_model=InfoResponse, tags=["rag"])
async def info(request: Request) -> InfoResponse:
    """Corpus size and available prompt versions."""
    service: RAGService = request.app.state.service
    return InfoResponse(
        documents_indexed=len(service.corpus.document_ids),
        chunks_indexed=len(service.corpus),
        prompt_versions=[{"version": v, "summary": s} for v, s in list_versions()],
        default_prompt_version=DEFAULT_VERSION,
    )


@router.post("/v1/ingest", response_model=IngestResult, tags=["rag"])
async def ingest(request: Request, payload: IngestRequest) -> IngestResult:
    """Chunk and index one document."""
    service: RAGService = request.app.state.service
    count = await service.ingest(payload.document_id, payload.text, title=payload.title)
    return IngestResult(document_id=payload.document_id, chunks_indexed=count)


@router.post("/v1/query", response_model=AnswerResult, tags=["rag"])
async def query(request: Request, payload: QueryRequest) -> AnswerResult:
    """Answer a question, with per-claim verification and abstention."""
    service: RAGService = request.app.state.service
    started = time.perf_counter()
    try:
        result = await service.query(payload)
    except TabulaRagError as exc:
        QUERIES.labels(outcome=exc.code).inc()
        raise
    QUERIES.labels(outcome="abstained" if result.abstained else "answered").inc()
    QUERY_LATENCY.observe(time.perf_counter() - started)
    return result
