#!/bin/bash
# Load every published layout of one RoboDojo task through the harness and report which ones fail
# to build or settle: the check the harness runs at each reset, where a failing layout is skipped.
# No policy is needed: a zero-action server stands in, and each episode takes a single step.
#   GPU=1 GROUP=0 scripts/termitech/layout_check.sh press_by_number
# GROUP is the Eval_Layout group, which the harness's `seed` param selects; the published
# protocol uses group 0. Pick a GPU nobody else is using (`nvidia-smi`).
set -euo pipefail
source "$(dirname "$0")/env.sh"
TASK=${1:?usage: layout_check.sh TASK}; GPU=${GPU:-1}; GROUP=${GROUP:-0}; PORT=${PORT:-$((18100 + GPU))}
N=$(ls "$ROBODOJO_ASSETS/Eval_Layout/RoboDojo/arx_x5/$GROUP" | grep -cE "^${TASK}_[0-9]+\.json$" || true)
[ "$N" -gt 0 ] || { echo "no layouts for $TASK in group $GROUP" >&2; exit 1; }
RUN=layouts_${TASK}_g${GROUP}_$(date +%Y%m%d_%H%M%S)
LOGS=$MEMVLA_DATA/logs/$RUN OUT=$MEMVLA_DATA/results/$RUN
RESULT=$OUT/RoboDojoBenchmark_aggregate.json
mkdir -p "$LOGS"
cd "$MEMVLA_CODE/vla-evaluation-harness"

ZERO_SERVER='
import sys
import anyio
import numpy as np
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve_async

class ZeroActions(PredictModelServer):
    def predict(self, obs, ctx):
        return {"actions": np.zeros(14, dtype=np.float32)}

anyio.run(serve_async, ZeroActions(), "127.0.0.1", int(sys.argv[1]))
'
CUDA_VISIBLE_DEVICES= setsid "$ROBODOJO_ENV/bin/python" -c "$ZERO_SERVER" "$PORT" > "$LOGS/server.log" 2>&1 &
SERVER=$! SIM=
trap 'kill -9 -- -"$SERVER" ${SIM:+-"$SIM"} 2>/dev/null || true' EXIT
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do kill -0 "$SERVER"; sleep 2; done

# One episode per layout. Isaac Sim crashes or hangs on shutdown (see smoke.sh), so wait for the
# results file rather than the process.
CUDA_VISIBLE_DEVICES=$GPU setsid micromamba run -p "$ROBODOJO_ENV" vla-eval run --no-docker \
    -c configs/benchmarks/robodojo/smoke_test.yaml --param "tasks=[$TASK]" --param "root=$ROBODOJO_ROOT" \
    --param "seed=$GROUP" --benchmark-field "episodes_per_task=$N" --benchmark-field max_steps=1 \
    --server-url "ws://127.0.0.1:$PORT" --output-dir "$OUT" > "$LOGS/sim.log" 2>&1 &
SIM=$!
until [ -f "$RESULT" ]; do kill -0 "$SIM" 2>/dev/null || break; sleep 10; done
for _ in $(seq 12); do kill -0 "$SIM" 2>/dev/null || break; sleep 5; done

python3 - "$TASK" "$GROUP" "$N" "$LOGS/sim.log" "$RESULT" <<'EOF'
import json
import re
import sys

task, group, n, log, result = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in open(log, errors="replace").read().splitlines()]
failed = {}
for i, line in enumerate(lines):
    m = re.search(r"RoboDojo layout (\d+) failed", line)
    if m:
        err = next((l.strip() for l in lines[i + 1:i + 120]
                    if re.match(r"^[\w.]+(Error|Exception)\b", l.strip())), "error not found in log")
        failed[int(m.group(1))] = err[:150]
built = set()
try:
    for t in json.load(open(result))["tasks"]:
        for e in t["episodes"]:
            layout = (e.get("metrics") or {}).get("layout_id")
            if isinstance(layout, int) and layout >= 0:
                built.add(layout)
except FileNotFoundError:
    print(f"no results file; see {log}")
untested = sorted(set(range(n)) - built - set(failed))
print(f"{task}, group {group}: {len(built)} of {n} layouts built and settled; "
      f"{len(failed)} failed; {len(untested)} not reached")
for layout, err in sorted(failed.items()):
    print(f"  layout {layout}: {err}")
if untested:
    print(f"  not reached: {untested}")
EOF
echo "results: $OUT   logs: $LOGS"
