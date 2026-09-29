# syntax=docker/dockerfile:1.7
#
# Packages the tabula-rag API service only. It calls out to three separately
# deployed vLLM servers on the MI300X (text generation, embeddings, and
# reranking — see docker-compose.yml's gpu profile and
# deploy/provision_mi300x.sh), which keeps this image's rebuilds fast and
# keeps the accelerator-facing serving stack independently upgradable.

FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="tabula-rag" \
      org.opencontainers.image.description="Grounded RAG over proprietary documents, with claim-level verification and abstention." \
      org.opencontainers.image.source="https://github.com/example/tabula-rag" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system --gid 1000 tabula \
    && useradd --system --uid 1000 --gid tabula --create-home tabula

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TABULA_RAG_ENVIRONMENT=prod \
    TABULA_RAG_LOG_FORMAT=json

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=tabula:tabula corpus ./corpus

USER tabula
EXPOSE 8090

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8090/healthz', timeout=2)" || exit 1

ENTRYPOINT ["uvicorn", "tabula_rag.api:create_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8090"]
CMD ["--workers", "2"]
