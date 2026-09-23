#!/bin/bash
#SBATCH --job-name=memvla-eval
#SBATCH --nodes=1
#SBATCH --gpus=1                  # one GH200: H100 96 GB + 72 Grace cores + ~115 GiB RAM
#SBATCH --time=04:00:00
#SBATCH --output=/projects/b5cs/memvla/logs/eval_%j.out   # Slurm cannot expand vars here; edit for another project
# RoboMME evaluation of one MME-VLA checkpoint through vla-evaluation-harness on one GH200.
#
#   sbatch eval.sbatch MODEL SUITE [TASKS] [EPISODES] [SHARDS] [EXTRA_RUN_ARGS...]
#     MODEL     config in configs/model_servers/mme_vla/: pi05_baseline, framesamp_modul, ttt_expert, ...
#     SUITE     config in configs/benchmarks/robomme/: counting, permanence, reference, imitation, eval
#     TASKS     optional subset, e.g. "[VideoUnmask,ButtonUnmask]" (default: the whole suite)
#     EPISODES  per task (default 50, the full RoboMME test split)
#     SHARDS    parallel (model server, simulator) pairs (default 4)
#     EXTRA_RUN_ARGS  passed to every `vla-eval run`, e.g. --param perturbation=shuffle
#   RUN_TAG (env) is appended to the results folder name, to tell conditions apart.
#
# Every shard gets its own model server: memory variants keep a single memory per server process,
# so shards sharing one server would overwrite each other's memory.
# Checkpoints must be downloaded and unpacked first (03_download_checkpoints.sh); jobs run offline.
#
# Quick end-to-end check (~5 min): sbatch --time=00:20:00 eval.sbatch pi05_baseline counting "[PickXtimes]" 2 1
set -euo pipefail
MODEL=${1:?model config name}; SUITE=${2:?suite config name}
TASKS=${3:-}; EPISODES=${4:-50}; SHARDS=${5:-4}; EXTRA_RUN_ARGS=("${@:6}")
source "${MEMVLA_REPO:-$HOME/code/MemVLA}/scripts/isambard/env.sh"
export HF_HUB_OFFLINE=1
cd "$MEMVLA_CODE/vla-evaluation-harness"
SIM_BIN=$MEMVLA_SIM_ENV/bin
NAME=${MODEL}_${SUITE}${RUN_TAG:+_$RUN_TAG}_${SLURM_JOB_ID}
# Straight to project storage: its Lustre is mounted with `flock`, so the shards can share one
# SQLite recording. The node-local $LOCALDIR is not safe for batch jobs here: its runtime
# directory can vanish mid-job, which failed job 6833513.
OUT=$MEMVLA_ROOT/results/$NAME
LOGS=$MEMVLA_ROOT/logs/$NAME
mkdir -p "$LOGS"
BASE_PORT=$((20000 + (SLURM_JOB_ID % 1000) * 16))   # the node may be shared with other jobs

# Serve the unpacked checkpoint rather than the Hugging Face id in the harness config: the
# harness's own unzip hides history_config.txt, so memory variants would fail to build.
MODEL_CFG=configs/model_servers/mme_vla/$MODEL.yaml
VARIANT=$(basename "$(awk '$1 == "checkpoint:" { print $2 }' "$MODEL_CFG")")
CKPT=$MEMVLA_ROOT/ckpts/$VARIANT/79999
[ -d "$CKPT/params" ] || { echo "no checkpoint at $CKPT: run 03_download_checkpoints.sh $VARIANT" >&2; exit 1; }

# Mesa's lavapipe keeps its memory in anonymous files under $XDG_RUNTIME_DIR. The per-user
# /run/user/<uid> behind it (and behind $LOCALDIR) can vanish during a batch job on these nodes,
# which killed jobs 6833513 and 6833653, so each job gets its own on node-local tmpfs.
export XDG_RUNTIME_DIR=/dev/shm/memvla-$SLURM_JOB_ID TMPDIR=/dev/shm/memvla-$SLURM_JOB_ID
mkdir -p -m 700 "$XDG_RUNTIME_DIR"

# Model servers on the GPU, one per shard, sharing its memory. Started one at a time: each load
# briefly holds the full weights in host RAM.
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(awk -v k="$SHARDS" 'BEGIN { printf "%.3f", 0.9 / k }')
servers=()
trap 'kill "${servers[@]}" 2>/dev/null || true; rm -rf "$XDG_RUNTIME_DIR"' EXIT
for i in $(seq 0 $((SHARDS - 1))); do
    uv run vla-eval serve -c "$MODEL_CFG" --arg "checkpoint=$CKPT" \
        --address "127.0.0.1:$((BASE_PORT + i))" > "$LOGS/server_$i.log" 2>&1 &
    servers+=($!)
    until curl -sf -o /dev/null "http://127.0.0.1:$((BASE_PORT + i))/health"; do
        kill -0 "${servers[$i]}"; sleep 10
    done
done
echo "$SHARDS servers healthy after ${SECONDS}s"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
free -g | head -2

# Simulators on the CPU, with the GPU hidden (why: scripts/isambard/README.md)
EVAL_ID=$(cat /proc/sys/kernel/random/uuid)
args=(--no-docker -c "configs/benchmarks/robomme/$SUITE.yaml" --benchmark-field "episodes_per_task=$EPISODES"
      --eval-id "$EVAL_ID" --num-shards "$SHARDS" --output-dir "$OUT" --record-video)
[ -z "$TASKS" ] || args+=(--param "tasks=$TASKS")
args+=("${EXTRA_RUN_ARGS[@]}")
shards=()
for i in $(seq 0 $((SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES= "$SIM_BIN/vla-eval" run "${args[@]}" --shard-id "$i" \
        --server-url "ws://127.0.0.1:$((BASE_PORT + i))" > "$LOGS/shard_$i.log" 2>&1 &
    shards+=($!)
done
failed=0
for p in "${shards[@]}"; do wait "$p" || failed=$((failed + 1)); done

"$SIM_BIN/vla-eval" export "$OUT/recording-$EVAL_ID.sqlite"
echo "results: $OUT  (elapsed ${SECONDS}s, $failed of $SHARDS shards failed)"
[ "$failed" -eq 0 ]
