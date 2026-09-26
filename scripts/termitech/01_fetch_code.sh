#!/bin/bash
# Clone the harness and RoboDojo into third_party/ at the commits vla-evaluation-harness v0.7.0
# validated RoboDojo against, and install uv and micromamba into $MEMVLA_DATA/bin. Safe to re-run.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$MEMVLA_CODE" "$MEMVLA_DATA"/{bin,envs,robodojo,results,logs} "$MEMVLA_CACHE"
cd "$MEMVLA_CODE"

clone_at() {  # <dir> <url> <tag or commit>
    local dir=$1 url=$2 rev=$3
    [ -d "$dir/.git" ] || git clone --quiet --filter=blob:none "$url" "$dir"
    git -C "$dir" fetch --quiet --tags origin
    git -C "$dir" switch --quiet -C memvla-base "$rev"
    echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

clone_at vla-evaluation-harness https://github.com/allenai/vla-evaluation-harness.git v0.7.0
# Our changes to the harness, as commits on memvla-base (see patches/harness/)
for patch in "$MEMVLA_REPO"/scripts/termitech/patches/harness/*.patch; do
    git -C vla-evaluation-harness -c user.name=memvla -c user.email=memvla@localhost am --quiet "$patch"
done
# The harness's pin (configs/benchmarks/robodojo/README.md). Submodules stay at the commits this
# RoboDojo commit records; upstream's install.sh uses --remote, which tracks moving branch heads.
clone_at RoboDojo https://github.com/RoboDojo-Benchmark/RoboDojo.git ee67a1468510da7624a089164402359f2afc72c8
git -C RoboDojo submodule update --quiet --init --recursive
git -C RoboDojo submodule status
# Our fixes to RoboDojo, as commits on top of the pin (see patches/robodojo/). clone_at resets
# memvla-base, so they are re-applied on every run.
for patch in "$MEMVLA_REPO"/scripts/termitech/patches/robodojo/*.patch; do
    git -C RoboDojo -c user.name=memvla -c user.email=memvla@localhost am --quiet "$patch"
done
git -C RoboDojo log --oneline ee67a1468510da7624a089164402359f2afc72c8..HEAD

# Tools, in user space
[ -x "$MEMVLA_DATA/bin/micromamba" ] || \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MEMVLA_DATA" bin/micromamba
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$MEMVLA_DATA/bin" UV_NO_MODIFY_PATH=1 sh   # leave shell profiles alone
micromamba --version && uv --version
