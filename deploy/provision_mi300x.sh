#!/usr/bin/env bash
# Provisions a single AMD Developer Cloud MI300X droplet to serve all three
# models tabula-rag needs -- text generation, embeddings, and reranking --
# as three vLLM containers sharing one GPU, then smoke-tests each.
#
# Prerequisites: an MI300X GPU Droplet from the ROCm base image, provisioned
# via the AMD AI Developer Program credit (https://developer.amd.com/ai-developer-program/).
#
# Usage:
#   ./deploy/provision_mi300x.sh <droplet-ip>

set -euo pipefail

HOST="${1:?usage: provision_mi300x.sh <droplet-ip>}"
SSH_USER="${TABULA_SSH_USER:-root}"
LLM_MODEL="${TABULA_RAG_LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
EMBED_MODEL="${TABULA_RAG_EMBEDDING_MODEL:-BAAI/bge-m3}"
RERANK_MODEL="${TABULA_RAG_RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"

echo "==> Provisioning ${HOST}: ${LLM_MODEL} + ${EMBED_MODEL} + ${RERANK_MODEL} on ROCm vLLM"

# shellcheck disable=SC2087
ssh -o StrictHostKeyChecking=accept-new "${SSH_USER}@${HOST}" bash -s <<REMOTE
set -euo pipefail
rocm-smi --showproductname || { echo "rocm-smi not found -- is this a ROCm image?" >&2; exit 1; }

for name in tabula-llm tabula-embeddings tabula-reranker; do
  docker rm -f "\$name" >/dev/null 2>&1 || true
done
docker pull rocm/vllm:latest

echo "--> Starting text-generation server (port 8000)"
docker run -d --name tabula-llm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --restart unless-stopped \
  rocm/vllm:latest \
  vllm serve "${LLM_MODEL}" --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.55 \
    --guided-decoding-backend xgrammar

echo "--> Starting embedding server (port 8001)"
docker run -d --name tabula-embeddings --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8g \
  --security-opt seccomp=unconfined --restart unless-stopped \
  rocm/vllm:latest \
  vllm serve "${EMBED_MODEL}" --host 0.0.0.0 --port 8001 \
    --task embed --gpu-memory-utilization 0.15

echo "--> Starting reranker server (port 8002)"
docker run -d --name tabula-reranker --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8g \
  --security-opt seccomp=unconfined --restart unless-stopped \
  rocm/vllm:latest \
  vllm serve "${RERANK_MODEL}" --host 0.0.0.0 --port 8002 \
    --task score --gpu-memory-utilization 0.15

echo "--> Waiting for all three to report healthy"
for port in 8000 8001 8002; do
  for _ in \$(seq 1 60); do
    curl -sf "http://127.0.0.1:\${port}/health" >/dev/null 2>&1 && { echo "    :\${port} healthy"; break; }
    sleep 5
  done
done
REMOTE

echo "==> Smoke test from this machine"
for port in 8000 8001 8002; do
  curl -sf "http://${HOST}:${port}/v1/models" | python3 -m json.tool || {
    echo "Smoke test failed for port ${port} -- check container logs on ${HOST}" >&2
    exit 1
  }
done

cat <<EOF

==> Ready. Point the API service at these endpoints:

    export TABULA_RAG_LLM_BASE_URL="http://${HOST}:8000/v1"
    export TABULA_RAG_EMBEDDING_BASE_URL="http://${HOST}:8001/v1"
    export TABULA_RAG_RERANKER_BASE_URL="http://${HOST}:8002/v1"

Record utilisation while you work:  ssh ${SSH_USER}@${HOST} 'rocm-smi --showuse --showmemuse'
Destroy the droplet when done -- credits bill hourly whether or not the GPU is in use.
EOF
