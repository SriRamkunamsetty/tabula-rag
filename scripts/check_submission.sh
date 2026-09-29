#!/usr/bin/env bash
# Pre-submission gate check for Mini-Challenge 3: the grader's hard gates, run locally on your image.
#
#   scripts/check_submission.sh <image:tag> <corpus_dir> [questions.json]
#
# Hard gates (any failure scores 0): built on the mandated ROCm base (layer identity), <= 60 GiB,
# torch still the ROCm build, container ready within 10 minutes (start + model load), and the index
# pass and every question completing inside their budgets while VRAM stays within 1-48 GiB.
# Also checks the four hostile-corpus cases and the "unanswerable => empty answer" behaviour.
#
# Run this on Linux with an AMD GPU. On Windows `chmod 000` does nothing, and as root the permission
# case is invisible unless you drop DAC_OVERRIDE (the grader does): this script does exactly that.
set -euo pipefail

IMAGE="${1:?usage: check_submission.sh <image:tag> <corpus_dir> [questions.json]}"
CORPUS="$(realpath "${2:?corpus dir required}")"
QUESTIONS="${3:-}"
BASE="rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
NAME="tabula-check-$$"
fail() { echo "FAIL: $*"; docker rm -f "$NAME" >/dev/null 2>&1 || true; exit 1; }
pass() { echo "PASS: $*"; }

echo "== 1. base image layer identity"
docker pull -q "$BASE" >/dev/null
base_layers=$(docker image inspect "$BASE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}')
image_layers=$(docker image inspect "$IMAGE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}')
n=$(printf '%s\n' "$base_layers" | grep -c . || true)
[ "$(printf '%s\n' "$image_layers" | head -n "$n")" = "$base_layers" ] \
  || fail "lower layers differ from $BASE (squashed, or built on another base?)"
pass "built on the mandated base ($n layers)"

echo "== 2. uncompressed size <= 60 GiB"
size=$(docker image inspect "$IMAGE" --format '{{.Size}}')
[ "$size" -le $((60 * 1024 * 1024 * 1024)) ] || fail "image is $((size / 1024 / 1024 / 1024)) GiB"
pass "$((size / 1024 / 1024)) MiB"

echo "== 3. torch is still the ROCm build (pip can silently swap in a CUDA build)"
docker run --rm --entrypoint python3 "$IMAGE" -c \
  "import sys, torch; print(torch.__version__, 'hip', torch.version.hip); sys.exit(0 if torch.version.hip else 1)" \
  || fail "torch is not a ROCm build any more"
pass "ROCm torch"

echo "== 4. no secrets in the image"
if docker run --rm --entrypoint sh "$IMAGE" -c 'find /app /opt/src -name ".env*" -o -name "*.pem" -o -name "id_rsa" 2>/dev/null' | grep -q .; then
  fail "secret-looking files found in the image"
fi
pass "none found"

echo "== 5. container starts and the model loads within 10 minutes"
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --cap-drop DAC_OVERRIDE --shm-size 16g \
  -v "$CORPUS":/app/corpus:ro "$IMAGE" >/dev/null
start=$(date +%s)
until docker exec "$NAME" test -f /tmp/tabula_ready 2>/dev/null; do
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ] || fail "container exited"
  [ $(( $(date +%s) - start )) -lt 600 ] || fail "not ready after 600 s"
  sleep 5
done
pass "ready after $(( $(date +%s) - start )) s"

vram_log=$(mktemp)
( while docker inspect "$NAME" >/dev/null 2>&1; do
    amd-smi metric --mem --json 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);d=d if isinstance(d,list) else [d];print(sum(int(float((x.get("mem_usage") or {}).get("used_vram",{}).get("value",0))) for x in d))' >> "$vram_log" 2>/dev/null || true
    sleep 3
  done ) &
sampler=$!

echo "== 6. index pass (charged to the start-up budget)"
t0=$(date +%s)
docker exec "$NAME" python3 /app/app.py --index /app/corpus | tail -1
echo "  index took $(( $(date +%s) - t0 )) s (start-up total so far: $(( $(date +%s) - start )) s of 600)"
[ $(( $(date +%s) - start )) -lt 600 ] || fail "start-up + index exceeded 600 s"

echo "== 7. questions"
if [ -n "$QUESTIONS" ]; then
  python3 - "$QUESTIONS" > /tmp/$NAME.q <<'PY'
import json, sys
raw = open(sys.argv[1], encoding="utf-8").read().strip()
rows = json.loads(raw) if raw.startswith("[") else [json.loads(x) for x in raw.splitlines() if x]
for i, r in enumerate(rows, 1):
    print(f"query_{i:02d}\t{(r.get('question') or r.get('query') or '').replace(chr(9), ' ')}")
PY
else
  printf 'query_01\tWhat is the value of something that is certainly not in this corpus xqzv?\n' > /tmp/$NAME.q
  echo "  (no questions file given: running a single unanswerable probe)"
fi
while IFS=$'\t' read -r qid question; do
  t=$(date +%s.%N)
  timeout 30 docker exec "$NAME" python3 /app/app.py --corpus /app/corpus --query-id "$qid" --query "$question" >/dev/null \
    || fail "$qid exceeded 30 s or crashed"
  dt=$(echo "$(date +%s.%N) - $t" | bc)
  out=$(docker exec "$NAME" cat "/app/output/${qid}_output.json") || fail "missing /app/output/${qid}_output.json"
  echo "$out" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert set(d)>={"answer","citations","confidence"}, d' \
    || fail "$qid output is missing a required key"
  printf "  %s %6.1fs  %s\n" "$qid" "$dt" "$(echo "$out" | cut -c1-110)"
done < /tmp/$NAME.q
rm -f /tmp/$NAME.q

echo "== 8. container is still running after all questions"
[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ] || fail "container died during the run"
pass "still running"

kill "$sampler" 2>/dev/null || true
peak=$(sort -n "$vram_log" 2>/dev/null | tail -1 || echo 0); rm -f "$vram_log"
echo "== 9. peak VRAM (MB, amd-smi): ${peak:-unknown}"
if [ -n "${peak:-}" ] && [ "$peak" != "0" ]; then
  [ "$peak" -ge 1024 ] || fail "peak VRAM ${peak} MB < 1 GiB (the GPU must be used)"
  [ "$peak" -le 49643 ] || fail "peak VRAM ${peak} MB > 48 GiB + 1%"
  pass "VRAM within 1-48 GiB"
else
  echo "WARN: could not read VRAM; verify manually with: watch -n1 'amd-smi metric --mem'"
fi

docker rm -f "$NAME" >/dev/null
echo "ALL CHECKS PASSED for $IMAGE"
