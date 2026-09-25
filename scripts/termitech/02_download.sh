#!/bin/bash
# RoboDojo assets (~90 GB) and the released π0.5 checkpoint (training seed 0, ~7 GB) from the
# RoboDojo-Benchmark/RoboDojo dataset (Apache-2.0, ungated), through the Hugging Face mirror.
# Same files as upstream's scripts/init_assets.sh, which needs git-lfs and huggingface.co.
# Resumable: re-run after an interruption.
set -euo pipefail
source "$(dirname "$0")/env.sh"
hf() { uvx --quiet --from huggingface_hub hf "$@"; }
dest=$MEMVLA_DATA/robodojo

hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset --local-dir "$dest" \
    --include "Assets/*" --max-workers 16
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset --local-dir "$dest" \
    --include "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/*"

for sub in Robots Object Material Eval_Layout; do     # init_assets.sh's completeness check
    [ -d "$ROBODOJO_ASSETS/$sub" ] || { echo "missing $ROBODOJO_ASSETS/$sub" >&2; exit 1; }
done
[ -d "$ROBODOJO_PI05_CKPT/params" ] || { echo "missing $ROBODOJO_PI05_CKPT/params" >&2; exit 1; }
ln -sfn "$ROBODOJO_ASSETS" "$ROBODOJO_ROOT/Assets"    # where RoboDojo's own code looks
du -sh --apparent-size "$ROBODOJO_ASSETS" "$(dirname "$ROBODOJO_PI05_CKPT")"
