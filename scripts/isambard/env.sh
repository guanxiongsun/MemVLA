# MemVLA on Isambard-AI. Source this in every shell and job:
#   source ~/code/MemVLA/scripts/isambard/env.sh
#
# Code is this git repo ($MEMVLA_REPO); the pinned upstream checkouts sit in third_party/ and
# are gitignored. Everything large — environments, checkpoints, results — stays on project
# storage ($MEMVLA_ROOT), and per-user caches on scratch, because $HOME is capped at 100 GiB.

export MEMVLA_REPO=${MEMVLA_REPO:-$HOME/code/MemVLA}
export MEMVLA_CODE=$MEMVLA_REPO/third_party
export MEMVLA_ROOT=$PROJECTDIR/memvla
export MEMVLA_CACHE=$SCRATCHDIR/memvla-cache

export HF_HOME=$MEMVLA_ROOT/hf                  # model checkpoints
export MS_ASSET_DIR=$MEMVLA_ROOT/maniskill      # ManiSkill assets
export UV_CACHE_DIR=$MEMVLA_CACHE/uv
export UV_LINK_MODE=copy                        # cache and envs sit on different filesystems
export UV_PYTHON_PREFERENCE=only-managed        # never build on the system Python 3.6 or miniconda
export VLA_EVAL_HOME=$MEMVLA_CACHE/vla-eval     # harness caches; assets/mme-vla links to our clone
export XDG_CACHE_HOME=$MEMVLA_CACHE/xdg
export JAX_COMPILATION_CACHE_DIR=$MEMVLA_CACHE/jax
export MAMBA_ROOT_PREFIX=$MEMVLA_ROOT/micromamba
export PATH=$MEMVLA_ROOT/bin:$HOME/.local/bin:$PATH

# Simulator env (RoboMME + ManiSkill + SAPIEN + harness client), built by 04_sim_env.sh
export MEMVLA_SIM_ENV=$MEMVLA_ROOT/envs/robomme
export MEMVLA_LAVAPIPE_ENV=$MEMVLA_ROOT/envs/lavapipe   # Mesa CPU Vulkan driver, kept separate
export ROBOMME_LAVAPIPE_ICD=$MEMVLA_LAVAPIPE_ENV/share/vulkan/icd.d/lvp_icd.aarch64.json
memvla_sim() {
    eval "$(micromamba shell hook -s bash)" && micromamba activate "$MEMVLA_SIM_ENV"
}
