#!/bin/bash
# The RoboDojo simulator env: Python 3.11, PyTorch 2.7 (cu128), Isaac Sim 5.1, Isaac Lab and cuRobo
# from RoboDojo's pinned submodules, plus the harness client. It follows RoboDojo's
# scripts/install.sh and Dockerfile step by step, in user space: conda-forge provides the system
# libraries the Docker image apt-installs (Vulkan loader, GLU, cmake, ninja, ffmpeg).
#
# Installing and running Isaac Sim means accepting NVIDIA's Isaac Sim EULA and Omniverse privacy
# terms (https://docs.omniverse.nvidia.com/eula/). Upstream's installer accepts them silently;
# this script refuses until you have read them and run
#   ACCEPT_NVIDIA_EULA=YES scripts/termitech/04_robodojo_env.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ "${ACCEPT_NVIDIA_EULA:-}" != "YES" ]; then
    echo "Read https://docs.omniverse.nvidia.com/eula/, then re-run with ACCEPT_NVIDIA_EULA=YES" >&2
    exit 1
fi
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TERM=xterm-256color
# cuRobo compiles CUDA kernels with the server's 12.4 toolkit (same major version as torch's cu128
# runtime, which is what torch's extension builder checks). A100 is sm_80.
export CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0

[ -x "$ROBODOJO_ENV/bin/python" ] || micromamba create -y -p "$ROBODOJO_ENV" --no-rc \
    -c conda-forge --override-channels python=3.11 pip libvulkan-loader libglu vulkan-tools cmake ninja ffmpeg
set +u; robodojo_env; set -u          # activation scripts are not nounset-clean
# Network from mainland China: PyPI's CDN gives this machine ~0.1 MB/s and Aliyun's PyPI mirror
# ~5 MB/s, so both uv and pip (Isaac Lab's installer calls pip) use the mirror. Isaac Sim's own
# wheels come from the local wheelhouse filled by 05_isaacsim_wheels.sh (NVIDIA's index is slower
# still). uv replaces pip for installs: same packages and pins, parallel downloads.
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export UV_HTTP_TIMEOUT=300 UV_INDEX_STRATEGY=unsafe-best-match
pip() { uv pip install --quiet --python "$ROBODOJO_ENV/bin/python" "$@"; }
WH=$MEMVLA_DATA/wheels/isaacsim-5.1.0
# All 25 wheels present and none still in flight (aria2c keeps a .aria2 file beside partial ones)
[ "$(ls "$WH"/*.whl 2>/dev/null | wc -l)" -eq 25 ] && ! ls "$WH"/*.aria2 >/dev/null 2>&1 \
    || { echo "Isaac Sim wheels incomplete: run 05_isaacsim_wheels.sh first" >&2; exit 1; }
pins=(numpy==1.26.0 packaging==23.0 typing_extensions==4.12.2 filelock==3.13.1 websockets==12.0
      click==8.1.7 psutil==5.9.8 wheel==0.45.1 starlette==0.45.3 scipy==1.15.3 warp-lang==1.11.0
      "onnx>=1.18,<1.22" "ipython<9" virtualenv==20.30.0)          # install.sh: pin_runtime_deps
cd "$ROBODOJO_ROOT"

# install.sh: setup_base_deps
pip -r scripts/requirements.txt
pip opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn numpy==1.26.0
# install.sh: setup_isaacsim
pip numpy==1.26.0 typing_extensions==4.12.2 filelock==3.13.1
pip torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip "$WH"/*.whl                                          # NVIDIA's wheels, verified, from disk
pip "isaacsim[all,extscache]==5.1.0" --find-links "$WH"   # confirms the set; the rest is PyPI
pip "${pins[@]}"
# install.sh: setup_isaaclab, with "none" as in the Dockerfile (eval needs no RL frameworks)
(cd third_party/IsaacLab && ./isaaclab.sh --install none)
pip "${pins[@]}"
# install.sh: setup_curobo
python -m pip uninstall -y nvidia-curobo curobo >/dev/null 2>&1 || true
constraints=$(mktemp -p "$MEMVLA_CACHE")
printf '%s\n' "${pins[@]}" 'stable-baselines3<2.8' > "$constraints"
(cd third_party/curobo && pip -e ".[cu12]" --no-build-isolation --constraint "$constraints")
rm -f "$constraints"
pip "${pins[@]}"
python -m pip uninstall -y python-discovery >/dev/null 2>&1 || true

# Harness client last, as the harness's RoboDojo image does (it lifts websockets to >=13)
pip -e "$MEMVLA_CODE/vla-evaluation-harness"

# Put only conda's Vulkan loader and GLU on the library path, not the whole env lib dir, and pin
# the driver's Vulkan ICD (the harness notes a second ICD makes Isaac fail to find the GPU).
mkdir -p "$ROBODOJO_ENV/syslibs" "$ROBODOJO_ENV/etc/conda/activate.d"
ln -sfn "$ROBODOJO_ENV/lib/libvulkan.so.1" "$ROBODOJO_ENV/syslibs/libvulkan.so.1"
ln -sfn "$ROBODOJO_ENV/lib/libGLU.so.1" "$ROBODOJO_ENV/syslibs/libGLU.so.1"
cat > "$ROBODOJO_ENV/etc/conda/activate.d/robodojo.sh" <<EOF
export LD_LIBRARY_PATH=$ROBODOJO_ENV/syslibs\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
EOF

python -c "import isaacsim, isaaclab, curobo, vla_eval, torch; print('imports ok; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
