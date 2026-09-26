#!/bin/bash
# Evaluate π0.5 on one RoboDojo task under the published protocol: that task's entry of the
# harness's configs/benchmarks/robodojo/eval.yaml (layout group, episode count, step limits), as
# scripts/run_robodojo_protocol.sh builds it. The π0.5 server runs on one GPU and Isaac Sim on
# another; pick GPUs nobody else is using (`nvidia-smi`). EPISODES overrides the protocol's count.
#   GPU_SIM=1 GPU_MODEL=3 EPISODES=3 scripts/termitech/eval_task.sh press_by_number
# Videos go to the results folder. Process handling follows smoke.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
TASK=${1:?usage: eval_task.sh TASK}; GPU_SIM=${GPU_SIM:-1}; GPU_MODEL=${GPU_MODEL:-3}
PORT=${PORT:-$((18000 + GPU_SIM))}
RUN=eval_${TASK}_$(date +%Y%m%d_%H%M%S)
LOGS=$MEMVLA_DATA/logs/$RUN OUT=$MEMVLA_DATA/results/$RUN
RESULT=$OUT/RoboDojoBenchmark_aggregate.json
mkdir -p "$LOGS"
cd "$MEMVLA_CODE/vla-evaluation-harness"

"$ROBODOJO_ENV/bin/python" - "$TASK" "$LOGS/config.yaml" <<'EOF'
import copy
import sys

import yaml

task, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/benchmarks/robodojo/eval.yaml"))
entry = next(e for e in cfg["benchmarks"] if task in e.get("params", {}).get("tasks", []))
entry = copy.deepcopy(entry)
entry["params"]["tasks"] = [task]
cfg["benchmarks"] = [entry]
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
EOF

CUDA_VISIBLE_DEVICES=$GPU_MODEL setsid uv run vla-eval serve -c configs/model_servers/robodojo_pi05/pi05.yaml \
    --address "127.0.0.1:$PORT" > "$LOGS/server.log" 2>&1 &
SERVER=$! SIM=
trap 'kill -9 -- -"$SERVER" ${SIM:+-"$SIM"} 2>/dev/null || true' EXIT
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do kill -0 "$SERVER"; sleep 10; done

CUDA_VISIBLE_DEVICES=$GPU_SIM setsid micromamba run -p "$ROBODOJO_ENV" vla-eval run --no-docker \
    -c "$LOGS/config.yaml" --param "root=$ROBODOJO_ROOT" ${EPISODES:+--benchmark-field "episodes_per_task=$EPISODES"} \
    --server-url "ws://127.0.0.1:$PORT" --output-dir "$OUT" --record-video > "$LOGS/sim.log" 2>&1 &
SIM=$!
until [ -f "$RESULT" ]; do kill -0 "$SIM" 2>/dev/null || break; sleep 30; done
for _ in $(seq 12); do kill -0 "$SIM" 2>/dev/null || break; sleep 5; done
[ -f "$RESULT" ] || { echo "no results; see $LOGS/sim.log" >&2; exit 1; }
"$ROBODOJO_ENV/bin/python" - "$RESULT" <<'EOF'
import json
import sys

episodes = [e for t in json.load(open(sys.argv[1]))["tasks"] for e in t["episodes"]]
for e in episodes:
    m = e.get("metrics") or {}
    print(f"layout {m.get('layout_id')}: {e.get('steps')} steps, success={m.get('success')}, score={m.get('score')}")
scores = [(e.get("metrics") or {}).get("score") or 0.0 for e in episodes]
wins = [bool((e.get("metrics") or {}).get("success")) for e in episodes]
if episodes:
    print(f"{len(episodes)} episodes: mean score {sum(scores) / len(scores):.3f}, success {sum(wins)}/{len(wins)}")
EOF
echo "results: $OUT   logs: $LOGS"
