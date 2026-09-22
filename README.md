# Who Moved It? Memory-augmented VLAs under partner-caused state changes

*Working title · Kick-off: week of 21 Sep 2026 · Target: CVPR 2027 (paper deadline 16 Nov 2026 AoE)*

**Quick links:** [CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) · [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) · [RoboMME repo](https://github.com/RoboMME/robomme_benchmark) · [RoboMME data](https://huggingface.co/datasets/lerobot/robomme) · [RoboMME paper](https://arxiv.org/abs/2603.04639) · [openpi](https://github.com/Physical-Intelligence/openpi) · [MEM](https://www.pi.website/research/memory) · [RoboMME-Interference](https://arxiv.org/abs/2606.22338) · [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo) · [Isambard docs](https://docs.isambard.ac.uk)

---

## 1. The idea

**In one sentence.** Memory-augmented VLAs are trained on single-agent demonstrations, so their memory may track what *the robot did* rather than what *changed in the world*, and should fail when the same change is made by someone else.

**Why this should be true.** In single-agent demonstration data, almost every state change is caused by the robot itself. A learned memory can therefore pass today's memory benchmarks by reading its own action history, proprioception or self-narrated notes instead of perceiving the change. Physical Intelligence's MEM makes the pattern concrete: its long-term memory is natural-language notes about the robot's own completed subtasks, and one of its comparison baselines is proprioceptive memory. A change the robot caused can always be recovered from its own history; a change someone else caused can only be recovered by perceiving it.

### Hypotheses

| ID | Hypothesis |
|---|---|
| **H1** agent gap | With the resulting world state held fixed, success drops when a partner, not the robot, causes the change. |
| **H2** visibility | The gap grows when the partner acts outside the robot's view. |
| **H3** shortcut | The gap is largest for memory that can be read off the robot's own history (symbolic or self-narrated notes, action/proprio history) and smallest for perception-grounded memory. |
| **H4** fix | Training with partner perturbations closes the gap without hurting single-agent performance. |

### Experimental design

- **Controlled variable:** who causes the change: `self` · `visible partner` · `environment` (no visible cause, as in RoboMME's Swap tasks).
- **Second factor:** the change happens `in view` or `out of view`.
- **Held fixed:** task, resulting world state, seeds (paired episodes across conditions).
- **Validity check:** every condition must leave visible evidence of the change somewhere in the observation stream; verify per episode.
- **Partner:** a scripted second arm in simulation; the second physical arm on the real setup. No human participants in this paper.
- **Metrics:** success rate with 95% Wilson CIs; agent gap Δ = SR(self) − SR(partner) with bootstrap CIs over paired seeds.

### Planned contributions (CVPR 2027)

1. A causal-agent extension of RoboMME with a scripted partner (to be released).
2. Evaluation of the released MME-VLA π0.5 memory variants plus a second model family.
3. One fix: **F1** partner-perturbation data augmentation, or **F2** an agent-agnostic event memory on frozen visual features.
4. Real-robot validation on two arms, one scripted as the partner.

### Positioning: what is already taken

| Work | Covers | Our angle |
|---|---|---|
| MEM, Physical Intelligence, Mar 2026 (arXiv 2603.03596) | Causal video encoder (short-term) + language notes (long-term) on π0.6; tasks up to 15 min | Notes narrate the robot's own subtasks; partner-caused changes not studied (as far as we can tell) |
| RoboMME + MME-VLA (arXiv 2603.04639) | 16 memory tasks; 14 π0.5 memory variants (symbolic / perceptual / recurrent × context / modulator / expert) | Single-agent; environment swaps in the Permanence suite. **Our base benchmark and baselines** |
| RoboMME-Interference (arXiv 2606.22338) | Unrelated sessions inserted into the memory history | Not a second agent. **Template for an extension paper** |
| AGM (arXiv 2608.29537) | Memory updates grounded in observed execution evidence; one rollout shows a human refilling cubes | **Closest neighbour: read in full.** Causal agent is not a controlled variable (as far as we can tell) |
| MIKASA-Robo (arXiv 2502.10550), RMBench (2603.01229), EventVLA (2606.20092), μVLA (2606.12497), MemoryVLA (2508.19236) | Memory benchmarks and mechanisms | Single-agent focus |
| Faithfulness in embodied reasoning (arXiv 2607.04681) | Counterfactual tests of VLA chain-of-thought | Per-step reasoning, not persistent memory |

CVPR 2027 treats papers that appeared online after **15 Sep 2026** as contemporaneous: cite and discuss them, but they are not grounds for rejection. Keep `docs/related_work.md` updated weekly.

---

## 2. Milestone 1: Stand up the harness and reproduce the baseline plus two variants

**Goal.** Run RoboMME with MME-VLA checkpoints through AllenAI's vla-evaluation-harness and show our numbers match the published ones. Every result in the paper goes through this pipeline, so this step is not optional.

**Platform.** Evaluation runs on Isambard-AI (aarch64 GH200). The environment is built and smoke-tested end to end (§2.0.1). Physics and rendering can differ between architectures, so every number in the paper comes from this one platform, and §2.5 checks our numbers against the published ones before we trust them.

**Definition of done**

- [ ] π0.5 baseline and two memory variants reproduce published per-task success within our 95% CI on at least 3 of the 5 tasks in §2.4, or each discrepancy is explained.
- [ ] Wall-clock time and GPU-hours per (task, model) evaluation recorded, giving the experiment budget for weeks 2–8.
- [ ] Swap vs non-Swap table committed (§2.6).
- [ ] Every run records platform, checkpoint and render mode in `docs/repro_log.md`; no results table mixes platforms.
- [ ] `docs/repro_log.md` complete.

### 2.0 Setup

Isambard-AI is set up and running baselines; §2.0.3 keeps an x86 host as an optional parity reference. Everything in §2.0.1 was executed and verified on 22 Sep 2026.

#### 2.0.1 Isambard-AI (ready)

The environment lives in `/projects/b5cs/memvla`, built by the scripts in [`scripts/isambard/`](scripts/isambard/), which are the record of what was installed and why. [`scripts/isambard/README.md`](scripts/isambard/README.md) has the layout, the commands and the ARM-specific workarounds.

```bash
clifton auth && ssh b5cs.aip2.isambard
cd ~/code/MemVLA && source scripts/isambard/env.sh
# end-to-end check, about 5 minutes
sbatch --time=00:20:00 scripts/isambard/eval.sbatch pi05_baseline counting "[PickXtimes]" 2 1
```

- This repo is checked out at `~/code/MemVLA`; the upstream baselines sit in `third_party/`, gitignored and pinned by `01_fetch_code.sh` to the commits harness v0.7.0 was validated against: harness `6cc3e1b`, RoboMME `f2b540e`, MME-VLA `ecf086c`, ManiSkill fork `07be6fb`, MPlib `v0.1.1`. Environments, checkpoints and results stay on project storage.
- Two Python environments: the simulator (SAPIEN 3.0.3, ManiSkill, RoboMME, mplib, harness client) and the model server built by uv (JAX 0.5.3 with CUDA 12, openpi fork).
- Checkpoints downloaded and unpacked: π0.5 baseline, FrameSamp+Modulator, TTT-Expert.
- Verified on a GH200: the baseline on `PickXtimes` and FrameSamp+Modulator on `VideoUnmask`, the latter with its conditioning video and memory active. The simulator runs at ~92 ms/step on the CPU renderer, ~174 ms/step with the model in the loop, so a 1300-step episode takes under 4 minutes.

Five pieces needed ARM-specific work, all done and each explained in the scripts' README: mplib compiled from source, SAPIEN installed from its GitHub release with its bundled glibc/GCC runtime libraries repointed at the system's, conda's pinocchio Python bindings moved aside, simulators run with the GPU hidden, and checkpoints unpacked so memory variants find their memory config.

#### 2.0.1a Reference: what makes this platform different

- **Access:** Isambard-AI Phase 2, project `b5cs`. Run `clifton auth` (the certificate lasts 12 h), then `ssh b5cs.aip2.isambard`.
- **Hardware:** aarch64 Grace CPUs, SLES 15 SP6. A node holds 4 GH200s. `--gpus=1` allocates one H100 (96 GB) plus 72 Grace cores and about 115 GiB of RAM. Jobs run for at most 24 h. Login nodes are for editing, downloads and builds; everything else goes through Slurm.
- **No Docker and no Charliecloud**, the only runtimes the harness can drive. Apptainer and podman-hpc are installed, but the harness's own RoboMME image is amd64-only anyway, so we install the stack natively (conda-forge plus uv) and run `vla-eval run --no-docker`.
- **Rendering** uses Mesa lavapipe on the CPU, as the harness's RoboMME configs default to. SAPIEN's native NVIDIA path is untested here and would need a re-run of every number.
- **Network:** login nodes reach GitHub, Hugging Face, PyPI and the registries. Caches are filled on the login node and jobs run offline (`HF_HUB_OFFLINE=1`).

| Component (version pinned by harness v0.7.0) | On aarch64 |
|---|---|
| `vla-eval` v0.7.0 | Pure Python |
| MME-VLA server: openpi fork @ `ecf086c`, JAX 0.5.3 + CUDA 12 | Wheels exist for jaxlib, jax-cuda12, tensorstore, cuDNN and NCCL. torch 2.7.1 falls back to the CPU wheel, which the JAX policy never uses |
| SAPIEN 3.0.3 | No PyPI wheel, but the [3.0.3 GitHub release](https://github.com/haosulab/SAPIEN/releases/tag/3.0.3) ships official `linux_aarch64` wheels |
| mplib 0.1.1 (ManiSkill's motion planner; RoboMME calls it on every reset to build the conditioning video) | No wheel or source release in any version, and the fork needs the 0.1.x API, so it is compiled from source |
| fast_kinematics 0.2.2 | Wheel exists, and the fork never imports it |
| torch 2.9.1 (simulator side) | Wheel exists, CPU-only, which is all the simulator needs |

#### 2.0.2 Storage (Isambard-AI)

Nothing is backed up, and project storage is deleted when the project ends. `$HOME` (100 GiB) holds none of it.

| Location | Holds |
|---|---|
| `~/code/MemVLA` | This repo, with the pinned upstream checkouts in `third_party/` (gitignored) |
| `$PROJECTDIR/memvla/` | `envs/`, `ckpts/` unpacked checkpoints, `hf/` downloads, `results/`, `logs/` |
| `$SCRATCHDIR/memvla-cache/` | Per-user uv, harness and JAX caches |
| `$LOCALDIR` | Per-job SQLite recording and videos, copied out at job end |

Budget about 24 GB per checkpoint variant: the zip plus its unpacked copy.

#### 2.0.3 x86 H100 host (optional parity reference)

- [ ] Only needed if our numbers disagree with the published ones (§2.5): running the same seeds on x86 with `render: cpu` separates an architecture effect from a renderer one.
- [ ] x86_64 node with an H100, Ubuntu 22.04 (the OS openpi is tested on), NVIDIA driver, Docker + NVIDIA Container Toolkit. Clusters without a Docker daemon: `--runtime charliecloud` (v0.6.0+, needs unprivileged user namespaces).

#### 2.0.4 Repo skeleton

- [ ] Create the repo skeleton:

```text
.
├── README.md
├── third_party/ # upstream checkouts at pinned commits (gitignored; 01_fetch_code.sh recreates them)
├── configs/     # our copies/overrides of harness configs
├── scripts/     # launch, eval and analysis helpers
│   └── isambard/  # the Isambard-AI environment: setup scripts, env.sh, eval.sbatch (see its README)
├── envs/        # partner-agent task extensions (Milestone 2)
├── results/     # summary tables (large raw outputs gitignored)
└── docs/
    ├── repro_log.md
    └── related_work.md
```

### 2.1 Install the harness (pinned)

Done on Isambard-AI: `scripts/isambard/01_fetch_code.sh` clones the harness at v0.7.0 (`6cc3e1b`, 19 Sep 2026) alongside RoboMME, MME-VLA, the ManiSkill fork and MPlib at their pinned commits, and `02_model_env.sh` builds the harness CLI and model-server environments. On any other host:

```bash
git submodule add https://github.com/allenai/vla-evaluation-harness.git third_party/vla-evaluation-harness
cd third_party/vla-evaluation-harness
git checkout v0.7.0   # latest stable at kick-off
uv sync --python 3.11 --all-extras --dev
```

- [ ] Record the tag and commit hash in `docs/repro_log.md`.

### 2.2 Locate configs and checkpoints

- [x] Configs (harness v0.7.0). Model servers are in `configs/model_servers/mme_vla/`: `pi05_baseline.yaml`, `framesamp_modul.yaml`, `ttt_expert.yaml`, `ttt_context.yaml`, plus the other variants. Benchmarks are in `configs/benchmarks/robomme/`: `counting.yaml`, `permanence.yaml`, and `eval.yaml` for all four suites. There is no smoke-test config; pass `--benchmark-field episodes_per_task=2` instead.
- [x] Released checkpoints (checked on Hugging Face, 21 Sep):
  - `Yinpei/pi05_baseline`.
  - In `Yinpei/mme_vla_suite`: FrameSamp × {Context, Modulator, Expert}, TokenDrop × {Context, Modulator, Expert}, TTT × {Context, Expert}, and Symbolic × {simple, grounded subgoal}. Symbolic variants read predefined subtask annotations.
  - Still empty: all RMT variants and TTT-Modulator. The harness ships configs for them anyway.
- [x] The harness's reproduction report, `docs/reproductions/robomme.md`:
  - Only the π0.5 baseline is reproduced, and only on Counting: 25.5% vs 22.72% reported, one seed, GPU rendering.
  - Memory variants are pending, and nothing has been run with CPU rendering.
  - Published numbers average 9 runs (3 checkpoints × 3 seeds), but Hugging Face has one checkpoint per variant, so expect extra variance.
  - Target numbers still come from the per-task tables in the RoboMME paper.
- [ ] Download only the variants you need (`hf download … --include "<variant>/*"`) and start the server with `HF_HUB_OFFLINE=1`. Otherwise `snapshot_download` pulls the whole 118.5 GB suite to run a single variant.
- [ ] Pin `mme_vla_suite`. The server imports it from a runtime clone of `robomme_policy_learning` at `main`, while openpi itself is pinned to `ecf086c`. Pre-clone at `ecf086c` as in §2.1, on any platform.
- [x] `chunk_size` settled: **16**, as shipped in `_base.yaml`. MME-VLA's own eval client executes 16 actions per inference (`examples/robomme/eval.py` passes `exec_horizon=obs_horizon`, and `utils.py` asserts it is 16). The "chunk_size=10" in the harness's report is the server script's default, not the value its config sets.

### 2.3 Smoke test (baseline)

Run these from `third_party/vla-evaluation-harness/`.

**x86 (Docker)**

- [ ] In the eval YAML set `docker.user: host`. Containers run as root by default, which leaves root-owned output and breaks `vla-eval export`.
- [ ] Terminal 1 (GPU host): start the model server and wait for HTTP 200 on `/health`.

```bash
uv run vla-eval serve --config configs/model_servers/mme_vla/pi05_baseline.yaml
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
```

- [ ] Terminal 2: run the smoke test.

```bash
uv run vla-eval run --config configs/benchmarks/robomme/counting.yaml --benchmark-field episodes_per_task=2 --record-video
```

**Isambard-AI (Slurm)** — done on 22 Sep 2026, and repeatable in about five minutes:

```bash
sbatch --time=00:20:00 scripts/isambard/eval.sbatch pi05_baseline counting "[PickXtimes]" 2 1
```

One job on one GH200 runs the model server on the GPU and the simulator on the CPU cores beside it. First run: the baseline scored 1 success in 2 `PickXtimes` episodes at ~174 ms/step, and FrameSamp+Modulator scored 1 in 2 on `VideoUnmask` with its memory active. Videos, per-episode JSONL and the aggregate JSON land in `$PROJECTDIR/memvla/results/<model>_<suite>_<jobid>/`.

- [ ] Watch two or three rollout videos before trusting a full run.

### 2.4 Reproduction set

| | Choice | Why |
|---|---|---|
| Tasks | `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`, `ButtonUnmaskSwap` (Permanence) + `PickXtimes` (Counting) | Swap pairs double as the free H1 check; one control from another suite |
| Models | π0.5 baseline · FrameSamp + Modulator · one TTT variant | No memory · best-balanced perceptual design in RoboMME · recurrent contrast |
| Episodes | RoboMME test split, 50 per task, fixed seeds | Matches the published protocol |

- [ ] Run 3 models × 5 tasks × 50 episodes. Per model that is one Permanence job plus `PickXtimes` from Counting:

  ```bash
  sbatch scripts/isambard/eval.sbatch pi05_baseline permanence "" 50 4
  sbatch scripts/isambard/eval.sbatch pi05_baseline counting "[PickXtimes]" 50 4
  ```

  At ~2 minutes per episode and 4 shards, expect roughly 1.5 h per model, so about 5 GPU-hours for the set.
- [ ] Stay on CPU rendering (the RoboMME default). SAPIEN's native NVIDIA path is ~5–10× faster where it works, but it is untested on Isambard and switching renderer invalidates every number measured before it.
- [ ] **Memory variants need one model server per shard.** The MME-VLA server keeps memory in a single policy object and resets it at every episode start, so shards sharing a server overwrite each other's memory. For the same reason, `max_batch_size > 1` is rejected when memory is on. `eval.sbatch` gives each shard its own server; the stateless π0.5 baseline could share one.
- [ ] Raise `SHARDS` (each GPU comes with 72 cores) until time per episode starts to climb. `scripts/run_sharded.sh` from the harness is not usable here: it points every shard at one server URL and cannot forward `--no-docker`. Note v0.7.0 has no `vla-eval merge`; shards share one SQLite recording and `eval.sbatch` runs `vla-eval export` at the end.

### 2.5 Acceptance check

- [ ] Per (task, model): success rate with 95% Wilson CI; mark ✅ if the published number falls inside our CI.
- [ ] Note the power limit: with 50 episodes the CI half-width reaches about ±14 points at 50% success. Fine for reproduction, too wide for the paper's main comparisons. Plan for 150+ paired episodes per condition (about ±8) and check whether extra seeds can be generated beyond the fixed test split.
- [ ] Log wall-clock and GPU-hours per run. On Isambard-AI: `sacct -j <jobid> -o JobID,Elapsed,AllocTRES%60`.

### 2.6 Free first look at H1

- [ ] For each model, compute Δ = SR(non-Swap) − SR(Swap) for both Permanence pairs, from our runs and from the RoboMME tables.
- [ ] Caveat: RoboMME's swaps have no visible agent and may simply be harder tasks. Treat this as a signal, not evidence.

### `docs/repro_log.md` template

```markdown
| Item | Value |
|---|---|
| Harness tag / commit | |
| MME-VLA checkpoint IDs | |
| Platform | x86_64 + Docker / Isambard-AI aarch64 + Apptainer |
| Image | Docker tag + digest / SIF sha256 |
| SAPIEN / mplib | version + source (PyPI, GitHub release, our aarch64 build) |
| GPU / driver / CUDA | |
| Render mode | cpu / gpu |
| Slurm job IDs | |
| Date | |

| Task | Model | Ours: SR (95% CI) | Published SR | Match | Wall-clock | GPU-h |
|---|---|---|---|---|---|---|
```

---

## 3. Future steps

### Timeline to CVPR 2027

| Weeks | Dates (2026) | Work | Gate / output |
|---|---|---|---|
| 1 | 21–27 Sep | Milestone 1. Isambard-AI environment built and smoke-tested on 22 Sep (§2.0.1); next is the reproduction set (§2.4). **In parallel:** get MME-VLA training running on an Isambard-AI GH200 (MME-VLA is JAX-based and its JAX stack has aarch64 CUDA wheels; openpi's PyTorch port would need the cu128 aarch64 torch wheel instead); real-robot fine-tuning pipeline on our own demos (LeRobot format); memory fit and control-loop latency of one variant on the RTX 5090; file the ethics application for the human-partner follow-up | `repro_log.md` |
| 2–3 | 28 Sep–11 Oct | Partner extension in 2–3 RoboMME tasks: scripted partner arm, cause × visibility conditions, matched end states, visible-evidence check | **Gate ~10 Oct: clear self-vs-partner gap on at least one memory variant, or stop the CVPR attempt** |
| 2–5 | 28 Sep–25 Oct | Real-robot track: partner-arm scripting, demos for 2 tasks, fine-tuning | **Gate ~25 Oct: policy and partner script running end-to-end on the arms, or real robot moves to supplementary** |
| 3–5 | 5–25 Oct | Full grid: baseline + released variants + second family (OpenVLA-OFT / MemoryVLA adaptations in the RoboMME repo) | Main results table |
| 4–6 | 12 Oct–1 Nov | Fix F1 and/or F2; retrain baseline + 2–3 variants; evaluate checkpoints during training via the harness Python API (v0.6.0+) | Fix results |
| 6–7 | 26 Oct–8 Nov | Real-robot evaluation: ~2 tasks × ~20 trials × 3 methods | Real-robot table + video |
| 7–8 | 2–15 Nov | Writing, ablations, supplementary video; second related-work sweep around 2 Nov | Full draft |
| — | 16 Nov | **Paper deadline (AoE).** Supplementary due 23 Nov AoE. Check the official call for the paper-registration date; third-party trackers disagree | Submission |

### Compute placement

| Resource | Use | Notes |
|---|---|---|
| Isambard-AI (GH200, aarch64) | Evaluation (set up, §2.0.1) and training | 4 × H100 96 GB per node, so full π0.5 fine-tuning (> 70 GB) fits on one GPU. No Docker; the stack is installed natively. Each GPU comes with 72 Grace cores for CPU-rendered sim shards. Jobs ≤ 24 h |
| Cloud H100 (x86) | Optional parity reference; training overflow | Full π0.5 fine-tuning needs > 70 GB; openpi is tested on Ubuntu 22.04 |
| A100 cluster | Parallel single-node runs (variants × seeds), eval shards | openpi's training script is single-node |
| RTX 5090 (32 GB) | Real-robot inference; LoRA fine-tuning (> 22.5 GB) | Check memory-variant fit and latency early |

### Risks

- **H1 is false** (no gap): stop at the 10 Oct gate and redirect to the next suitable 2027 venue.
- **Augmentation closes the gap trivially:** the paper becomes diagnosis plus a simple fix; still viable if the diagnosis is sharp.
- **Getting scooped:** this area moves in weeks. Run a weekly arXiv sweep and log it in `docs/related_work.md`.
- **Statistical power:** the 50-episode test split is too small for the main claims.
- **Tooling:** checkpoint availability, renderer quirks, JAX/PyTorch mismatch between MME-VLA and our code.
- **ARM platform:** the environment works, but as far as we know nobody has published RoboMME numbers from aarch64, or from CPU rendering for memory variants. Physics and rendering can drift between platforms, so no table mixes them, and §2.5 compares against the published numbers before we build on ours. Rebuilding the stack elsewhere means redoing the five workarounds in `scripts/isambard/README.md`.

### After CVPR

- Replace the scripted partner with a human: dyadic assembly data, egocentric + exocentric views, partner-intention inference, leading to a collaborator-conditioned VLA.
- Build a sim twin of the real setup: straightforward if the arms are DROID-style Franka (Isaac Lab Arena, PolaRiS); otherwise it costs a URDF, camera calibration and sim demos.
