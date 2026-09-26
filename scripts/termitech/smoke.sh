#!/bin/bash
# One short RoboDojo episode through the harness: the π0.5 server on one GPU, Isaac Sim on another
# (rendering and inference sharing a GPU is fragile). There is no scheduler on this machine, so
# pick GPUs nobody else is using (`nvidia-smi`).
#   GPU_SIM=1 GPU_MODEL=3 scripts/termitech/smoke.sh [TASK]
#
# Isaac Sim crashes or hangs while shutting down, after the results are written (the harness's
# RoboDojo notes report the same), so this waits for the results file rather than the process,
# then stops it. Server and simulator run in their own process groups, so stopping them frees the
# GPUs completely; killing only `uv run` left the server holding 61 GB.
set -euo pipefail
source "$(dirname "$0")/env.sh"
TASK=${1:-stack_blocks}; GPU_SIM=${GPU_SIM:-1}; GPU_MODEL=${GPU_MODEL:-3}; PORT=${PORT:-18000}
RUN=smoke_${TASK}_$(date +%Y%m%d_%H%M%S)
LOGS=$MEMVLA_DATA/logs/$RUN OUT=$MEMVLA_DATA/results/$RUN
RESULT=$OUT/RoboDojoBenchmark_aggregate.json
mkdir -p "$LOGS"
cd "$MEMVLA_CODE/vla-evaluation-harness"

CUDA_VISIBLE_DEVICES=$GPU_MODEL setsid uv run vla-eval serve -c configs/model_servers/robodojo_pi05/pi05.yaml \
    --address "127.0.0.1:$PORT" > "$LOGS/server.log" 2>&1 &
SERVER=$! SIM=
trap 'kill -9 -- -"$SERVER" ${SIM:+-"$SIM"} 2>/dev/null || true' EXIT
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do kill -0 "$SERVER"; sleep 10; done
echo "model server healthy after ${SECONDS}s"

CUDA_VISIBLE_DEVICES=$GPU_SIM setsid micromamba run -p "$ROBODOJO_ENV" vla-eval run --no-docker \
    -c configs/benchmarks/robodojo/smoke_test.yaml --param "tasks=[$TASK]" --param "root=$ROBODOJO_ROOT" \
    --server-url "ws://127.0.0.1:$PORT" --output-dir "$OUT" --record-video > "$LOGS/sim.log" 2>&1 &
SIM=$!
until [ -f "$RESULT" ]; do kill -0 "$SIM" 2>/dev/null || break; sleep 10; done
for _ in $(seq 12); do kill -0 "$SIM" 2>/dev/null || break; sleep 5; done   # 60 s to exit cleanly
[ -f "$RESULT" ] || { echo "no results; see $LOGS/sim.log" >&2; exit 1; }
python3 -c 'import json, sys
e = json.load(open(sys.argv[1]))["tasks"][0]["episodes"][0]
print("episode:", e["name"], e["steps"], "steps", e["metrics"])' "$RESULT"
echo "results: $OUT   logs: $LOGS"
