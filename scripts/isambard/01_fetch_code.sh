#!/bin/bash
# Clone the upstream baselines into third_party/ at the commits vla-evaluation-harness v0.7.0
# was validated with. Each repo gets a local branch `memvla-base` at the pinned commit; the
# checkouts are gitignored, and these pins are the record. Safe to re-run.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$MEMVLA_CODE" "$MEMVLA_ROOT"/{bin,ckpts,envs,hf,logs,maniskill,results,wheels} "$MEMVLA_CACHE"
cd "$MEMVLA_CODE"

clone_at() {  # <dir> <url> <tag or commit>
    local dir=$1 url=$2 rev=$3
    [ -d "$dir/.git" ] || git clone --quiet --filter=blob:none "$url" "$dir"
    git -C "$dir" fetch --quiet --tags origin
    git -C "$dir" switch --quiet -C memvla-base "$rev"
    echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

clone_at vla-evaluation-harness  https://github.com/allenai/vla-evaluation-harness.git  v0.7.0
clone_at robomme_benchmark       https://github.com/RoboMME/robomme_benchmark.git       f2b540e64034cf9afd0b51cf8ec491de09c74f58
clone_at robomme_policy_learning https://github.com/RoboMME/robomme_policy_learning.git ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
clone_at ManiSkill               https://github.com/YinpeiDai/ManiSkill.git              07be6fbc66350ddca200abfb0a11b692f078f7fd
clone_at MPlib                   https://github.com/haosulab/MPlib.git                  v0.1.1
git -C MPlib submodule update --quiet --init --recursive

# The model server imports mme_vla_suite from $VLA_EVAL_HOME/assets/mme-vla and would
# otherwise clone robomme_policy_learning@main at runtime; point it at the pinned clone.
mkdir -p "$VLA_EVAL_HOME/assets"
ln -sfn "$MEMVLA_CODE/robomme_policy_learning" "$VLA_EVAL_HOME/assets/mme-vla"
