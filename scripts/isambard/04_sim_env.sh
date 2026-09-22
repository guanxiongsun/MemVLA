#!/bin/bash
# Simulator env: RoboMME + ManiSkill fork + SAPIEN + the harness client, at $MEMVLA_SIM_ENV.
# Nothing here has an aarch64 wheel on PyPI except the Python-only parts, so:
#   - conda-forge supplies mplib's C++ deps at the versions mplib 0.1.1 was built with,
#     plus Mesa lavapipe (CPU Vulkan) for rendering;
#   - SAPIEN comes from its GitHub release (official linux_aarch64 wheel);
#   - mplib 0.1.1 is compiled here (no aarch64 wheel or sdist exists).
#
# Steps, in order:
#   04_sim_env.sh create                                              # login node
#   srun --gpus=1 --time=00:45:00 scripts/isambard/04_sim_env.sh mplib # compute node (compiles)
#   04_sim_env.sh install                                             # login node
set -euo pipefail
source "$(dirname "$0")/env.sh"

SAPIEN_WHEEL=https://github.com/haosulab/SAPIEN/releases/download/3.0.3/sapien-3.0.3-cp311-cp311-linux_aarch64.whl
PY=$MEMVLA_SIM_ENV/bin/python
upip() { uv pip install --python "$PY" "$@"; }

case "${1:-}" in
create)
    micromamba create -y -p "$MEMVLA_SIM_ENV" --no-rc -c conda-forge --override-channels \
        python=3.11 pip \
        pinocchio=2.6.21 ompl=1.6.0 fcl=0.7.0 urdfdom=4.0.0 orocos-kdl eigen=3.4.0 octomap=1.9.8 \
        "assimp>=5.3,<6" libboost-devel \
        c-compiler cxx-compiler cmake make pkg-config libvulkan-loader patchelf
    # Lavapipe gets its own env: its LLVM needs a newer ICU than pinocchio 2.6.21's boost allows.
    # Vulkan only needs the ICD path ($ROBOMME_LAVAPIPE_ICD), as in the harness's x86 images.
    micromamba create -y -p "$MEMVLA_LAVAPIPE_ENV" --no-rc -c conda-forge --override-channels \
        mesa-lavapipe=26.2.1
    # Build tools for mplib, which is built without isolation (see the mplib step)
    upip "setuptools<81" "setuptools-git-versioning<2" wheel build numpy"<2"
    ;;
mplib)
    memvla_sim
    cd "$MEMVLA_CODE/MPlib"
    # The wheel version comes from the git tag: a dirty tree would give 0.1.1.devYYYYMMDD, which
    # ManiSkill's `mplib==0.1.1` pin rejects. So start clean, and never edit tracked files.
    git checkout -- . && git status --short --untracked-files=no | { ! grep -q .; }
    # Keep the docstring headers shipped with v0.1.1. Regenerating them (dev/mkdoc.py) needs
    # upstream's build image, where every dependency header sits in /usr/local/include; here it
    # produces partial headers that break the bindings. A python3 shim skips just that step.
    shim=$(mktemp -d -p "$MEMVLA_CACHE")
    printf '#!/bin/sh\n[ "$1" = dev/mkdoc.py ] && exit 0\nexec "%s" "$@"\n' "$PY" > "$shim/python3"
    chmod +x "$shim/python3"
    export PATH=$shim:$PATH CMAKE_PREFIX_PATH=$CONDA_PREFIX
    export CMAKE_POLICY_VERSION_MINIMUM=3.5   # CMake 4 rejects the bundled pybind11's old minimum
    # mplib hard-codes -Werror into its Release flags, and GCC 15 raises new maybe-uninitialized
    # warnings inside pinocchio 2.6 headers. Build as RelWithDebInfo with the same -O3 instead.
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O3"
    export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_ON_NODE:-8}
    rm -rf build dist
    "$PY" -m build --wheel --no-isolation --skip-dependency-check --outdir "$MEMVLA_ROOT/wheels"
    rm -rf "$shim"
    git status --short --untracked-files=no   # prints nothing if the tree stayed clean
    ls -la "$MEMVLA_ROOT"/wheels/mplib-*.whl
    ;;
install)
    memvla_sim   # conda compilers on PATH: toppra builds from source
    # Same resolution cut-off as the harness's RoboMME image; --no-sources stops RoboMME's
    # tool.uv.sources from re-fetching ManiSkill from git over our local checkout.
    upip --exclude-newer 2026-07-07 --no-sources --find-links "$MEMVLA_ROOT/wheels" \
        "$SAPIEN_WHEEL" mplib==0.1.1 "torch==2.9.1" "torchvision==0.24.1" \
        -e "$MEMVLA_CODE/ManiSkill" \
        -e "$MEMVLA_CODE/robomme_benchmark" \
        -e "$MEMVLA_CODE/vla-evaluation-harness" \
        "setuptools==80.9.0" "opencv-python>=4.11.0.86" h5py imageio
    # SAPIEN's aarch64 wheel bundles runtime libraries from its manylinux_2_28 build image:
    # glibc 2.28's librt (calls that glibc's private symbols, so it fails to load on Isambard's
    # glibc 2.38) and GCC 8's libstdc++/libgcc_s (torch then segfaults if imported after sapien).
    # Link the system copies instead; they are newer and backward compatible.
    site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    for f in $(find "$site/sapien" "$site/sapien.libs" -name '*.so*' -type f); do
        for old in $(patchelf --print-needed "$f" | grep -E '^(librt-2-|libstdc\+\+-|libgcc_s-)' || true); do
            case $old in
                librt-*) new=librt.so.1 ;;
                libstdc++-*) new=libstdc++.so.6 ;;
                libgcc_s-*) new=libgcc_s.so.1 ;;
            esac
            patchelf --replace-needed "$old" "$new" "$f"
        done
    done
    # conda's pinocchio 2.6.21 is here as a C++ library for mplib, but it also ships Python
    # bindings. SAPIEN imports `pinocchio` when it can find it and then segfaults on the clash
    # with its own bundled C++ runtime; pip installs have no such module and SAPIEN uses its own
    # code. Move the bindings aside so this env behaves the same.
    mkdir -p "$MEMVLA_SIM_ENV/share/memvla-disabled"
    for m in pinocchio hppfcl eigenpy; do
        if [ -d "$site/$m" ]; then
            rm -rf "$MEMVLA_SIM_ENV/share/memvla-disabled/$m"
            mv "$site/$m" "$MEMVLA_SIM_ENV/share/memvla-disabled/"
        fi
    done
    VK_ICD_FILENAMES=$ROBOMME_LAVAPIPE_ICD "$PY" -c \
        'import sapien, mplib, mani_skill, robomme, vla_eval; print("imports ok: sapien", sapien.__version__)'
    ;;
*)
    echo "usage: $0 create | mplib | install" >&2; exit 2 ;;
esac
