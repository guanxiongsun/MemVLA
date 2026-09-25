#!/bin/bash
# Python envs for the harness CLI (vla-eval) and the RoboDojo π0.5 model server (XPolicyLab's
# openpi fork, JAX with CUDA 12). Download and install only; no Isaac Sim involved.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MEMVLA_CODE/vla-evaluation-harness"

uv sync --python 3.11 --all-extras --dev
# Builds the env `vla-eval serve` runs the server in (`uv run <script>` reuses it)
uv sync --script src/vla_eval/model_servers/robodojo_pi05.py
