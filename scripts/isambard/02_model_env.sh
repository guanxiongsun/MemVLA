#!/bin/bash
# Python envs for the harness CLI (vla-eval) and the MME-VLA model server
# (RoboMME openpi fork + JAX 0.5.3 with CUDA 12). Login node is fine: download and install only.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MEMVLA_CODE/vla-evaluation-harness"

uv sync --python 3.11 --all-extras --dev
# Builds the env that `vla-eval serve` later runs the server in (`uv run <script>` reuses it)
uv sync --script src/vla_eval/model_servers/mme_vla.py
