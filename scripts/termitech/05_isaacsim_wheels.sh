#!/bin/bash
# Isaac Sim 5.1.0's NVIDIA-hosted wheels (25 files, 4.7 GB) into a local wheelhouse, each checked
# against the SHA-256 pinned in isaacsim-5.1.0-wheels.txt. 04_robodojo_env.sh installs from it.
#
# Why a separate step: only NVIDIA's index has these wheels, and from mainland China it redirects to
# pypi.nvidia.cn, which gives this machine ~0.05 MB/s per connection and ~0.25 MB/s in total. pip and
# uv fetch each wheel on one connection with no resume, so the 3 GB wheel would never finish. aria2c
# splits each file over 16 connections, retries without limit and resumes. Expect ~5 hours; re-run
# to resume after an interruption.
set -euo pipefail
source "$(dirname "$0")/env.sh"
WH=$MEMVLA_DATA/wheels/isaacsim-5.1.0
TOOLS=$MEMVLA_DATA/envs/tools
mkdir -p "$WH"
[ -x "$TOOLS/bin/aria2c" ] || micromamba create -y -q -p "$TOOLS" --no-rc -c conda-forge --override-channels aria2

"$TOOLS/bin/aria2c" --input-file="$(dirname "$0")/isaacsim-5.1.0-wheels.txt" --dir="$WH" \
    --continue=true --check-integrity=true --max-concurrent-downloads=2 \
    --split=16 --max-connection-per-server=16 --min-split-size=1M \
    --max-tries=0 --retry-wait=15 --timeout=60 --connect-timeout=30 --file-allocation=none \
    --summary-interval=300 --console-log-level=warn
echo "wheels in $WH: $(ls "$WH"/*.whl | wc -l) of 25"
