# MemVLA on Isambard-AI. Source this in every shell and job:
#   source /projects/b5cs/memvla/scripts/isambard/env.sh
#
# Layout: code, envs, checkpoints and results live under $MEMVLA_ROOT (shared with the
# project); download caches are per-user under $SCRATCHDIR. Nothing goes in $HOME (100 GiB).

export MEMVLA_ROOT=$PROJECTDIR/memvla
export MEMVLA_CODE=$MEMVLA_ROOT/code
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
