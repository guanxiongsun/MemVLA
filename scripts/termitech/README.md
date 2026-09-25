# RoboDojo on the termitech A100 server

RoboDojo runs on Isaac Sim 5.1, whose renderer needs RT cores. Isambard's GH200s (H100) have none,
and the harness authors saw the renderer crash the GPU there, so RoboDojo runs on this x86 machine
instead: Ubuntu 22.04, 8× A100 80GB, driver 580. The harness authors found A100 works, although
NVIDIA does not officially support it.

```bash
ssh termitech
cd ~/code/MemVLA && source scripts/termitech/env.sh
```

## Layout

| Path | Contents |
|---|---|
| `~/code/MemVLA` | this repo |
| `third_party/vla-evaluation-harness` | harness v0.7.0 (`6cc3e1b`), same as on Isambard |
| `third_party/RoboDojo` | RoboDojo `ee67a14`, the harness's pin, with submodules at their recorded commits: IsaacLab `afca7b0`, cuRobo `d17b54c`, XPolicyLab `432f82b` |
| `$MEMVLA_DATA/envs/robodojo` | simulator env: Python 3.11, torch 2.7 (cu128), Isaac Sim 5.1, Isaac Lab, cuRobo, harness client |
| `$MEMVLA_DATA/robodojo/Assets` | RoboDojo assets, 86 GB, dataset revision `43dacb1` |
| `$MEMVLA_DATA/robodojo/ckpt/.../RoboDojo-sim-arx_x5-joint-0/59999` | released π0.5 checkpoint, training seed 0 |
| `$MEMVLA_DATA/{results,logs}` | outputs |
| `$MEMVLA_DATA/cache/uv` | uv cache, including the π0.5 model-server env (JAX, XPolicyLab's openpi fork) |

`MEMVLA_DATA` is `/data/sgx/memvla-data` on the 5 TB data disk (`/data/sgx` is private to this
account, like the other users' folders there); `~/memvla-data` is a symlink to it, so paths
recorded before the move still work. Set `MEMVLA_DATA` before sourcing `env.sh` to use another
location.

## Steps

```bash
scripts/termitech/01_fetch_code.sh      # upstream code at the pinned commits; uv, micromamba
scripts/termitech/02_download.sh        # assets + π0.5 checkpoint, 93 GB, resumable
scripts/termitech/03_model_env.sh       # harness CLI and π0.5 server envs
ACCEPT_NVIDIA_EULA=YES scripts/termitech/04_robodojo_env.sh   # only after reading NVIDIA's EULA
GPU_SIM=1 GPU_MODEL=3 scripts/termitech/smoke.sh               # one short episode end to end
```

Run long steps detached, since there is no scheduler here:
`setsid nohup <script> > $MEMVLA_DATA/logs/<name>.log 2>&1 < /dev/null &`.

## Differences from RoboDojo's own install

- **No Docker, no root.** The machine is shared and has no Docker, so the Docker image's apt
  packages come from conda-forge inside the env: the Vulkan loader, GLU, cmake, ninja and ffmpeg.
  Only the Vulkan loader and GLU are put on `LD_LIBRARY_PATH`, and Vulkan is pinned to the
  driver's `/usr/share/vulkan/icd.d/nvidia_icd.json`.
- **Pinned submodules.** Upstream's `install.sh` updates submodules with `--remote`, which follows
  moving branch heads; these scripts keep the commits RoboDojo `ee67a14` records.
- **Hugging Face through its mirror.** `huggingface.co` is blocked from this machine and
  `hf-mirror.com` is not. `hf download --include` never finishes here, because the mirror's
  paginated listing links back to huggingface.co, so `02_download.sh` lists files in one call and
  fetches them one by one. The mirror rate-limits bursts (HTTP 429); the downloader backs off and
  resumes.
- **EULA.** Installing and running Isaac Sim means accepting NVIDIA's Isaac Sim EULA and
  Omniverse privacy terms (https://docs.omniverse.nvidia.com/eula/). Upstream accepts them
  silently; `04_robodojo_env.sh` refuses until `ACCEPT_NVIDIA_EULA=YES` is given.

## Running evaluations

`smoke.sh` shows the pattern: the π0.5 server on one GPU, Isaac Sim on another, the harness in
`--no-docker` mode with `--param root=$ROBODOJO_ROOT`. Per the harness's RoboDojo notes: one task
per process (Isaac's simulation context is process-global), one simulator per GPU (sharing a GPU
made throughput ~8× worse), and roughly 12–20 GPU-hours per task at 50 episodes.

Check `nvidia-smi` before choosing GPUs: other users' processes may be running.
