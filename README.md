# Who Moved It? Memory-augmented VLAs under partner-caused state changes

*Working title · Kick-off: week of 21 Sep 2026 · Target: CVPR 2027 (paper deadline 16 Nov 2026 AoE)*

**Quick links:** [CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) · [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) · [RoboMME repo](https://github.com/RoboMME/robomme_benchmark) · [RoboMME data](https://huggingface.co/datasets/lerobot/robomme) · [RoboMME paper](https://arxiv.org/abs/2603.04639) · [openpi](https://github.com/Physical-Intelligence/openpi) · [MEM](https://www.pi.website/research/memory) · [RoboMME-Interference](https://arxiv.org/abs/2606.22338) · [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)

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

**Definition of done**

- [ ] π0.5 baseline and two memory variants reproduce published per-task success within our 95% CI on at least 3 of the 5 tasks in §2.4, or each discrepancy is explained.
- [ ] Wall-clock time and GPU-hours per (task, model) evaluation recorded, giving the experiment budget for weeks 2–8.
- [ ] Swap vs non-Swap table committed (§2.6).
- [ ] `docs/repro_log.md` complete.

### 2.0 Setup

- [ ] x86_64 node with an H100, Ubuntu 22.04 (the OS openpi is tested on), NVIDIA driver, Docker + NVIDIA Container Toolkit. Clusters without a Docker daemon: the harness supports Charliecloud from v0.6.0.
- [ ] `uv` installed. Disk: budget 100 GB+ if you pull every released checkpoint (π0.5-sized checkpoints are roughly 10 GB each), plus the benchmark image.
- [ ] Create the repo skeleton:

```text
.
├── README.md
├── third_party/vla-evaluation-harness   # git submodule, pinned tag
├── configs/     # our copies/overrides of harness configs
├── scripts/     # launch, eval and analysis helpers
├── envs/        # partner-agent task extensions (Milestone 2)
├── results/     # summary tables (large raw outputs gitignored)
└── docs/
    ├── repro_log.md
    └── related_work.md
```

### 2.1 Install the harness (pinned)

```bash
git submodule add https://github.com/allenai/vla-evaluation-harness.git third_party/vla-evaluation-harness
cd third_party/vla-evaluation-harness
git checkout v0.7.0   # latest stable at kick-off
uv sync --python 3.11 --all-extras --dev
```

- [ ] Record the tag and commit hash in `docs/repro_log.md`.

### 2.2 Locate configs and checkpoints

- [ ] Find the MME-VLA model-server config(s) under `configs/model_servers/` and the RoboMME benchmark config(s) under `configs/benchmarks/` (inside the harness). *Filenames in §2.3 are placeholders, not yet verified.*
- [ ] Confirm which MME-VLA checkpoints are released today. Last known (RoboMME-Interference, mid-2026): π0.5 baseline; FrameSamp × {Context, Modulator, Expert}; TokenDrop × {Context, Modulator, Expert}; TTT × {Context, Expert}. Not released then: all RMT variants and TTT-Modulator. Symbolic variants read predefined subtask annotations.
- [ ] Check whether the harness ships a reproduction report for MME-VLA; otherwise take target numbers from the per-task tables in the RoboMME paper.

### 2.3 Smoke test (baseline)

Run these from `third_party/vla-evaluation-harness/`.

- [ ] In the eval YAML set `docker.user: host`. Containers run as root by default, which leaves root-owned output and breaks `vla-eval merge`.
- [ ] Terminal 1 (GPU host): start the model server and wait for HTTP 200 on `/health`.

```bash
uv run vla-eval serve --config configs/model_servers/<mme_vla_pi05_baseline>.yaml
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:<port>/health
```

- [ ] Terminal 2: run the smoke test.

```bash
uv run vla-eval run --config configs/benchmarks/<robomme_smoke_test>.yaml --record-video
```

- [ ] Confirm result JSON appears in `results/` and watch two or three rollout videos.

### 2.4 Reproduction set

| | Choice | Why |
|---|---|---|
| Tasks | `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`, `ButtonUnmaskSwap` (Permanence) + `PickXtimes` (Counting) | Swap pairs double as the free H1 check; one control from another suite |
| Models | π0.5 baseline · FrameSamp + Modulator · one TTT variant | No memory · best-balanced perceptual design in RoboMME · recurrent contrast |
| Episodes | RoboMME test split, 50 per task, fixed seeds | Matches the published protocol |

- [ ] Run 3 models × 5 tasks × 50 episodes.
- [ ] Start with CPU rendering (the RoboMME default). Once numbers match, try `--render gpu` (roughly 5–10× faster on hosts where it works) and confirm the numbers don't move.
- [ ] If throughput is the bottleneck: shard with `scripts/run_sharded.sh` (then `vla-eval merge`) and enable model-server batching via `max_batch_size`.

### 2.5 Acceptance check

- [ ] Per (task, model): success rate with 95% Wilson CI; mark ✅ if the published number falls inside our CI.
- [ ] Note the power limit: with 50 episodes the CI half-width reaches about ±14 points at 50% success. Fine for reproduction, too wide for the paper's main comparisons. Plan for 150+ paired episodes per condition (about ±8) and check whether extra seeds can be generated beyond the fixed test split.
- [ ] Log wall-clock and GPU-hours per run.

### 2.6 Free first look at H1

- [ ] For each model, compute Δ = SR(non-Swap) − SR(Swap) for both Permanence pairs, from our runs and from the RoboMME tables.
- [ ] Caveat: RoboMME's swaps have no visible agent and may simply be harder tasks. Treat this as a signal, not evidence.

### `docs/repro_log.md` template

```markdown
| Item | Value |
|---|---|
| Harness tag / commit | |
| MME-VLA checkpoint IDs | |
| GPU / driver / CUDA | |
| Render mode | cpu / gpu |
| Date | |

| Task | Model | Ours: SR (95% CI) | Published SR | Match | Wall-clock | GPU-h |
|---|---|---|---|---|---|---|
```

---

## 3. Future steps

### Timeline to CVPR 2027

| Weeks | Dates (2026) | Work | Gate / output |
|---|---|---|---|
| 1 | 21–27 Sep | Milestone 1. **In parallel:** get MME-VLA training running on an H100 (confirm JAX vs openpi's PyTorch port before writing code); real-robot fine-tuning pipeline on our own demos (LeRobot format); memory fit and control-loop latency of one variant on the RTX 5090; file the ethics application for the human-partner follow-up | `repro_log.md` |
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
| Cloud H100 (x86) | Harness, training | Full π0.5 fine-tuning needs > 70 GB; openpi is tested on Ubuntu 22.04 |
| A100 cluster | Parallel single-node runs (variants × seeds), eval shards | openpi's training script is single-node |
| Isambard-AI (Grace Hopper, ARM) | Off the critical path until an ARM container is proven | |
| RTX 5090 (32 GB) | Real-robot inference; LoRA fine-tuning (> 22.5 GB) | Check memory-variant fit and latency early |

### Risks

- **H1 is false** (no gap): stop at the 10 Oct gate and redirect to the next suitable 2027 venue.
- **Augmentation closes the gap trivially:** the paper becomes diagnosis plus a simple fix; still viable if the diagnosis is sharp.
- **Getting scooped:** this area moves in weeks. Run a weekly arXiv sweep and log it in `docs/related_work.md`.
- **Statistical power:** the 50-episode test split is too small for the main claims.
- **Tooling:** checkpoint availability, renderer quirks, JAX/PyTorch mismatch between MME-VLA and our code.

### After CVPR

- Replace the scripted partner with a human: dyadic assembly data, egocentric + exocentric views, partner-intention inference, leading to a collaborator-conditioned VLA.
- Build a sim twin of the real setup: straightforward if the arms are DROID-style Franka (Isaac Lab Arena, PolaRiS); otherwise it costs a URDF, camera calibration and sim demos.
