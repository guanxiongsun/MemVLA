# MemVLA baselines on Isambard-AI

Everything lives under `/projects/b5cs/memvla` (shared with the project); per-user caches sit in
`$SCRATCHDIR/memvla-cache`, and nothing is written to `$HOME`. Start a session with:

```bash
clifton auth && ssh b5cs.aip2.isambard         # the certificate lasts 12 h
source /projects/b5cs/memvla/scripts/isambard/env.sh
```

## What is installed

| Path under `/projects/b5cs/memvla` | Contents |
|---|---|
| `code/vla-evaluation-harness` | AllenAI eval harness, tag v0.7.0 (`6cc3e1b`) |
| `code/robomme_benchmark` | RoboMME benchmark, `f2b540e` — the commit the harness validated against |
| `code/robomme_policy_learning` | MME-VLA policies (openpi fork), `ecf086c` — the harness's pin |
| `code/ManiSkill` | RoboMME's ManiSkill fork, `07be6fb` |
| `code/MPlib` | motion planner, tag v0.1.1, compiled here for ARM |
| `envs/robomme` | simulator env (Python 3.11): SAPIEN 3.0.3, ManiSkill, RoboMME, mplib, harness client |
| `envs/lavapipe` | Mesa 26.2.1 software Vulkan, used for rendering |
| `ckpts/<variant>/79999` | unpacked checkpoints: `pi05_baseline`, `perceptual-framesamp-modul`, `recurrent-ttt-expert` |
| `hf/` | Hugging Face cache holding the downloaded checkpoint zips |
| `results/`, `logs/` | evaluation outputs and job logs |

Each repo sits on a local branch `memvla-base` at the pinned commit. The model-server env
(JAX 0.5.3 with CUDA 12, plus the openpi fork) is built by uv and cached under
`$SCRATCHDIR/memvla-cache/uv`; `vla-eval serve` picks it up automatically.

## Running an evaluation

```bash
sbatch scripts/isambard/eval.sbatch MODEL SUITE [TASKS] [EPISODES] [SHARDS]

# quick end-to-end check, about 5 minutes
sbatch --time=00:20:00 scripts/isambard/eval.sbatch pi05_baseline counting "[PickXtimes]" 2 1

# one model over the Permanence suite: 4 tasks x 50 episodes, the published protocol
sbatch scripts/isambard/eval.sbatch framesamp_modul permanence "" 50 4
```

`MODEL` is any config in `code/vla-evaluation-harness/configs/model_servers/mme_vla/`, and
`SUITE` is `counting`, `permanence`, `reference`, `imitation` or `eval` (all four). Results land
in `results/<model>_<suite>_<jobid>/` as an aggregate JSON, per-episode JSONL, videos and a
SQLite recording; logs go to `logs/<model>_<suite>_<jobid>/`.

Every shard runs its own model server, because a memory variant keeps one memory per server
process and shards sharing a server would overwrite each other's memory. One GH200 holds four
pairs comfortably (each server preallocates `0.9 / SHARDS` of the 96 GB).

Add another variant with, for example:

```bash
scripts/isambard/03_download_checkpoints.sh perceptual-tokendrop-modul
```

Only `recurrent-rmt-*` and `recurrent-ttt-modul` are unavailable: those folders are empty on
Hugging Face.

## Rebuilding from scratch

```bash
scripts/isambard/01_fetch_code.sh                                   # login node
scripts/isambard/02_model_env.sh                                    # login node
scripts/isambard/03_download_checkpoints.sh                         # login node, ~36 GB
scripts/isambard/04_sim_env.sh create                               # login node
srun --gpus=1 --time=00:45:00 scripts/isambard/04_sim_env.sh mplib  # compiles, ~70 s
scripts/isambard/04_sim_env.sh install                              # login node
```

Simulator-only check (no model), useful after touching the sim env:

```bash
srun --gpus=1 --time=00:15:00 bash -c \
  'CUDA_VISIBLE_DEVICES= $MEMVLA_SIM_ENV/bin/python scripts/isambard/check_sim.py PickXtimes 0'
```

## Why the scripts do unusual things

This stack is built for x86 and none of it ships ARM builds, so the scripts work around five
things. Each was diagnosed on the cluster, not guessed:

- **mplib** has no aarch64 wheel or source release, so it is compiled here against conda-forge
  builds of the exact C++ versions it expects (pinocchio 2.6.21, OMPL 1.6.0, FCL 0.7.0, ...).
  Its docstring step is skipped: regenerating it needs upstream's build image, and the headers
  it ships are already correct.
- **SAPIEN**'s aarch64 wheel comes from its GitHub release (PyPI has none) and bundles glibc
  2.28's `librt` plus GCC 8's `libstdc++`/`libgcc_s`. These are repointed at the system copies,
  otherwise SAPIEN cannot load at all and torch segfaults when imported after it.
- **conda's pinocchio Python bindings** are moved aside. They exist only because mplib needs
  pinocchio's C++ library, and SAPIEN crashes if it finds them (a pip install never has them).
- **Simulators run with `CUDA_VISIBLE_DEVICES=`**, as the harness's x86 images do for CPU
  rendering. Otherwise ManiSkill selects a CUDA render device and calls `torch.cuda`, which the
  CPU-only aarch64 torch does not have.
- **Checkpoints are unpacked** so `history_config.txt` sits beside `79999/`, which is how
  MME-VLA's loader finds a variant's memory config. The harness's own unzip buries the
  checkpoint too deep for that, so memory variants fail to build through the stock path.

## Verified on 22 Sep 2026

| Check | Result |
|---|---|
| Simulator alone (PickXtimes) | reset 4.9 s, 92 ms/step with CPU rendering |
| π0.5 baseline, PickXtimes, 2 episodes | 1 success, ~174 ms/step end to end (job 6796071) |
| FrameSamp+Modulator, VideoUnmask, 2 episodes | 1 success, memory active (job 6796463) |

Reproduction against the published numbers has not been run yet; see §2.4 of the project README.
