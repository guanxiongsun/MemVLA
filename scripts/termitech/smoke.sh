#!/bin/bash
# One short RoboDojo episode through the harness: the π0.5 server on one GPU, Isaac Sim on another
# (rendering and inference sharing a GPU is fragile). There is no scheduler on this machine, so
# pick GPUs nobody else is using (`nvidia-smi`).
#   GPU_SIM=1 GPU_MODEL=3 scripts/termitech/smoke.sh [TASK]
set -euo pipefail
source "$(dirname "$0")/env.sh"
TASK=${1:-stack_blocks}; GPU_SIM=${GPU_SIM:-1}; GPU_MODEL=${GPU_MODEL:-3}; PORT=${PORT:-18000}
RUN=smoke_${TASK}_$(date +%Y%m%d_%H%M%S)
LOGS=$MEMVLA_DATA/logs/$RUN
mkdir -p "$LOGS"
cd "$MEMVLA_CODE/vla-evaluation-harness"

CUDA_VISIBLE_DEVICES=$GPU_MODEL uv run vla-eval serve -c configs/model_servers/robodojo_pi05/pi05.yaml \
    --address "127.0.0.1:$PORT" > "$LOGS/server.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do kill -0 "$SERVER"; sleep 10; done
echo "model server healthy after ${SECONDS}s"

CUDA_VISIBLE_DEVICES=$GPU_SIM micromamba run -p "$ROBODOJO_ENV" vla-eval run --no-docker \
    -c configs/benchmarks/robodojo/smoke_test.yaml --param "tasks=[$TASK]" --param "root=$ROBODOJO_ROOT" \
    --server-url "ws://127.0.0.1:$PORT" --output-dir "$MEMVLA_DATA/results/$RUN" --record-video \
    2>&1 | tee "$LOGS/sim.log"
echo "results: $MEMVLA_DATA/results/$RUN   logs: $LOGS"
