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
| `third_party/RoboDojo` | RoboDojo `ee67a14`, the harness's pin, plus our fixes from `patches/robodojo/` (see [Fixes to RoboDojo](#fixes-to-robodojo)), with submodules at their recorded commits: IsaacLab `afca7b0`, cuRobo `d17b54c`, XPolicyLab `432f82b` |
| `$MEMVLA_DATA/envs/robodojo` | simulator env: Python 3.11, torch 2.7 (cu128), Isaac Sim 5.1, Isaac Lab, cuRobo, harness client |
| `$MEMVLA_DATA/robodojo/Assets` | RoboDojo assets, 41 GB, dataset revision `43dacb1` |
| `$MEMVLA_DATA/robodojo/ckpt/.../RoboDojo-sim-arx_x5-joint-0/59999` | released π0.5 checkpoint, training seed 0: `params` 12 GB, plus `train_state` 32 GB (optimizer state, only needed to resume training) |
| `$MEMVLA_DATA/{results,logs}` | outputs |
| `$MEMVLA_DATA/cache/uv` | uv cache, including the π0.5 model-server env (JAX, XPolicyLab's openpi fork) |

`MEMVLA_DATA` is `/data/sgx/memvla-data` on the 5 TB data disk (`/data/sgx` is private to this
account, like the other users' folders there); `~/memvla-data` is a symlink to it, so paths
recorded before the move still work. Set `MEMVLA_DATA` before sourcing `env.sh` to use another
location.

## Steps

```bash
scripts/termitech/01_fetch_code.sh      # upstream code at the pinned commits, our RoboDojo fixes; uv, micromamba
scripts/termitech/02_download.sh        # assets + π0.5 checkpoint, 86 GB, resumable
scripts/termitech/03_model_env.sh       # harness CLI and π0.5 server envs
scripts/termitech/05_isaacsim_wheels.sh # Isaac Sim's NVIDIA-only wheels, 4.7 GB, ~5 h (see below)
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
- **Download sources, measured from this machine** (mainland China). Domestic CDNs are fast and
  international ones crawl, so each download goes where it is fastest:

  | Source | Speed | Used for |
  |---|---|---|
  | ModelScope | ~12 MB/s over 16 connections | RoboDojo assets and checkpoint (same dataset, hash-checked against the pinned Hugging Face revision) |
  | Aliyun PyPI mirror | ~5 MB/s | PyPI packages in the Isaac Sim env |
  | hf-mirror.com | ~2.5 MB/s, rate-limited | fallback for the assets; Hugging Face API calls |
  | download.pytorch.org | ~2.3 MB/s per connection | PyTorch cu128 |
  | PyPI's own CDN | ~0.1 MB/s | avoided |
  | NVIDIA's index (redirects to pypi.nvidia.cn) | ~0.25 MB/s in total | Isaac Sim's 25 wheels, via `aria2c` in `05_isaacsim_wheels.sh` |

  `huggingface.co` itself is blocked. Its mirror cannot run `hf download --include` here (the
  paginated listing links back to huggingface.co), so the file list comes from one API call.
  Isaac Sim's wheels exist only on NVIDIA's index; PyPI holds 1 KB placeholders. They are pinned
  with their SHA-256 in `isaacsim-5.1.0-wheels.txt`.
- **EULA.** Installing and running Isaac Sim means accepting NVIDIA's Isaac Sim EULA and
  Omniverse privacy terms (https://docs.omniverse.nvidia.com/eula/). Upstream accepts them
  silently; `04_robodojo_env.sh` refuses until `ACCEPT_NVIDIA_EULA=YES` is given.

## The two conflicts `pip check` reports

Both come from upstream pins that contradict each other, so no set of versions satisfies them all.
Neither matters for evaluation. This was checked on 26 Sep 2026 by running the smoke test with
`PYTHONVERBOSE=1`, which logs every module the simulator loads.

| Package | Installed | Conflicting requirements | Used during evaluation |
|---|---|---|---|
| starlette | 0.45.3: RoboDojo's pin, and the version Isaac Sim bundles | Isaac Lab wants `==0.49.1` (listed under "livestream"); Isaac Sim pins fastapi 0.115.7, which needs `<0.46` | never imported |
| websockets | 17.1 | Isaac Sim's kernel wants `==12.0`; the harness needs `>=13.0` | 17.1, by the harness client only |

- Only Isaac Sim's web-service extensions use starlette, and none of them start in a headless
  evaluation. The simulator opens no listening ports. 0.45.3 lacks denial-of-service fixes from
  later releases, but those only matter for a web server reachable from the network.
- Kit carries its own copies (websockets 12.0, starlette 0.45.3) in its `omni.kit.pip_archive`
  extension and puts them on `sys.path` once Isaac Sim starts. The harness imports websockets
  before starting Isaac Sim, so it keeps 17.1. If a harness update moved that import after Isaac
  Sim starts, the client would get 12.0. For Behavior1K, the harness authors deleted Kit's copy to
  avoid this.
- The π0.5 server runs in its own environment with websockets 16.1. Client and server only need
  to speak the WebSocket protocol, not share a library version.

## Fixes to RoboDojo

`01_fetch_code.sh` applies the patches in `patches/robodojo/` to RoboDojo `ee67a14`, as commits on
its `memvla-base` branch. They fix four bugs that only show when one process resets the scene many
times. The harness does exactly that, before every episode; RoboDojo's own evaluation runs 10
environments side by side, so a 50-episode task resets only five times.

| Patch | Bug | Without it |
|---|---|---|
| `0001` Reuse articulations across layouts | Articulated objects are built once, with the scene, but every later layout gave its articulation entries new names that no object has. | In every task with an articulated object, each layout after the first fails to build. This is why the harness authors found `press_by_number` and `swap_blocks` (whose buttons are articulated) broken. |
| `0002` Move deleted objects offscreen | A finished layout's objects are hidden and lose gravity, but stay where they are, with their collision shapes on. | Every task collects invisible obstacles wherever earlier episodes left their objects. Objects placed there are pushed aside, so layouts fail the stability check, and the robot can run into leftovers mid-episode. |
| `0003` Delete the previous room on reset | Rooms are deleted from `<env>/Room` but built under `<env>/Rooms`. | Each reset stacks another copy of the room. |
| `0004` Build the camera views once | Each reset added a render product per camera, never released, that every later render drew. | Rendering and observations slow down with every reset. |

Measured on 26 Sep 2026, layout group 0, one process per task:

- `press_by_number`: without the fixes, layout 0 built and layouts 1–38 all failed, until the
  harness's watchdog stopped the process. With them, all 65 layouts build and settle.
- `swap_blocks`: with only `0001`, 14 of the first 21 layouts failed the stability check; the two
  of those tried in a fresh process settled there. With all four fixes, all 55 layouts settle.
- Reset speed: with `0001` and `0002` only, a render took 40 ms at the first layout and 274 ms at
  the sixth, and a reset 18 s at the second layout and 33 s at the sixth. With all four, a render
  takes 40 ms throughout, and a reset about 15 s from the second layout to the 65th.

`layout_check.sh` repeats the first two checks and `reset_timing.py` the third.

What remains:

- Tasks whose articulated model changes between layouts (`make_toast`, `make_toast_random`,
  `store_laptop_and_headphones_random`) can only use layouts with the model of the process's first
  layout; the others fail with an error saying so and are skipped (by design, not yet tried).
- A finished layout's objects stay on the stage, hidden and 100 km away, 90–100 prims per
  reset. Deleting a rigid object mid-simulation invalidates Isaac Sim's physics views, after which
  the robot's state can no longer be read. The leftovers no longer slow rendering.
- The harness authors' RoboDojo reproduction (the four other Memory tasks, 15 episodes each) ran
  without these fixes.

## Running evaluations

`eval_task.sh` runs one task under the published protocol (the task's entry in the harness's
`configs/benchmarks/robodojo/eval.yaml`), with π0.5 on one GPU and Isaac Sim on another, and
records videos. `EPISODES` runs fewer episodes than the protocol's. `layout_check.sh` builds every
layout of a task without a policy.

```bash
GPU_SIM=1 GPU_MODEL=3 scripts/termitech/eval_task.sh press_by_number           # 50 episodes
GPU_SIM=1 GPU_MODEL=3 EPISODES=3 scripts/termitech/eval_task.sh swap_blocks
GPU=1 GROUP=0 scripts/termitech/layout_check.sh swap_blocks
```

Per the harness's RoboDojo notes: one task per process (Isaac's simulation context is
process-global), one simulator per GPU (sharing a GPU made throughput ~8× worse), and roughly 12–20
GPU-hours per task at 50 episodes, a figure measured without the fixes above.

Verified on 26 Sep 2026: `stack_blocks`, 20 steps, results and video written, with RTX-rendered
camera frames on the A100 (GPU 1 for Isaac Sim, GPU 3 for the server). After the fixes, π0.5 ran
3 protocol episodes each of `press_by_number` and `swap_blocks` (layouts 0–2, one process per
task) without errors, at 65–80 steps a minute; all six scored 0. The videos show each layout's own
number cards and cube positions, and the layout-1 `press_by_number` episode ends as the policy
presses the blue confirm button before the red presses the cards ask for.

Two things any run script must handle:

- **Isaac Sim crashes or hangs on shutdown**, after the results are written. Wait for the results
  file, not the process, then stop it; `smoke.sh` does this.
- **Stop whole process groups.** `vla-eval serve` starts the model under `uv run`; killing only the
  top process left the server holding 61 GB of GPU memory. Start each with `setsid` and kill its
  group.

On a crash, Isaac Sim's crash reporter uploads the dump to NVIDIA, as covered by the privacy terms
accepted for this setup. Pass `--/crashreporter/enabled=false` to Isaac Sim to turn it off.

Check `nvidia-smi` before choosing GPUs: other users' processes may be running.
