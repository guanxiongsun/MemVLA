# MemVLA on the termitech server (x86_64, Ubuntu 22.04, 8x A100 80GB, driver 580). Source it in
# every shell and job:
#   source ~/code/MemVLA/scripts/termitech/env.sh
#
# This machine runs RoboDojo (Isaac Sim 5.1), which Isambard cannot: its GH200s lack RT cores.
# Code is this repo, with upstream checkouts in third_party/ (gitignored). Everything large —
# environments, assets, checkpoints, results — lives under $MEMVLA_DATA. Nothing here needs root.

export MEMVLA_REPO=${MEMVLA_REPO:-$HOME/code/MemVLA}
export MEMVLA_CODE=$MEMVLA_REPO/third_party
export MEMVLA_DATA=${MEMVLA_DATA:-/data/$USER/memvla-data}   # the data disk; ~/memvla-data links here
export MEMVLA_CACHE=$MEMVLA_DATA/cache

# The server is in mainland China: huggingface.co is blocked, its mirror is reachable
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$MEMVLA_DATA/hf
export UV_CACHE_DIR=$MEMVLA_CACHE/uv UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed
export VLA_EVAL_HOME=$MEMVLA_CACHE/vla-eval XDG_CACHE_HOME=$MEMVLA_CACHE/xdg
export MAMBA_ROOT_PREFIX=$MEMVLA_DATA/micromamba
export PATH=$MEMVLA_DATA/bin:$HOME/.local/bin:$PATH

# RoboDojo
export ROBODOJO_ROOT=$MEMVLA_CODE/RoboDojo
export ROBODOJO_ENV=$MEMVLA_DATA/envs/robodojo            # Isaac Sim 5.1 + Isaac Lab + cuRobo + harness
export ROBODOJO_ASSETS=$MEMVLA_DATA/robodojo/Assets       # ~90 GB, Apache-2.0
export ROBODOJO_PI05_CKPT=$MEMVLA_DATA/robodojo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999

robodojo_env() {
    eval "$(micromamba shell hook -s bash)" && micromamba activate "$ROBODOJO_ENV"
}
