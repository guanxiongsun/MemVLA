#!/bin/bash
# Submit the memory-perturbation grid: one job per (task, condition), each on the same test
# episodes. `none_rep` reruns the clean condition to measure run-to-run noise.
#
#   experiments/memory_probes/submit.sh                       # the full grid, 50 episodes each
#   EPISODES=2 SHARDS=1 TIME=00:20:00 TASKS=VideoUnmaskSwap CONDITIONS=shuffle_kept \
#       experiments/memory_probes/submit.sh                   # smoke test
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/isambard/env.sh

MODEL=${MODEL:-framesamp_modul}
TASKS=${TASKS:-"VideoUnmask VideoUnmaskSwap"}
CONDITIONS=${CONDITIONS:-"none none_rep blank shuffle shuffle_kept reverse"}
EPISODES=${EPISODES:-50}
SHARDS=${SHARDS:-4}
TIME=${TIME:-02:00:00}

for task in $TASKS; do
    for cond in $CONDITIONS; do
        job=$(sbatch --parsable --time="$TIME" --job-name="probe-$cond" \
            --export="ALL,RUN_TAG=${task}_${cond},PYTHONPATH=$PWD/experiments/memory_probes" \
            scripts/isambard/eval.sbatch "$MODEL" permanence "[$task]" "$EPISODES" "$SHARDS" \
            --benchmark-field benchmark=perturbed_robomme:PerturbedRoboMMEBenchmark \
            --param "perturbation=${cond%_rep}")
        echo "$job  $MODEL  $task  $cond"
    done
done
