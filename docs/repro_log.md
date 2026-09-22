# Reproduction log

## Run 1 — π0.5 baseline, Isambard-AI, 22 Sep 2026

| Item | Value |
|---|---|
| Harness tag / commit | v0.7.0 / `6cc3e1b` |
| Checkpoint | `Yinpei/pi05_baseline`, step 79999 |
| Model config | `configs/model_servers/mme_vla/pi05_baseline.yaml`, `chunk_size` 16 |
| Benchmark | RoboMME `f2b540e`, ManiSkill fork `07be6fb`, SAPIEN 3.0.3, mplib 0.1.1 (built for aarch64) |
| Platform | Isambard-AI Phase 2, aarch64 GH200 (H100 96 GB), driver 565.57.01 |
| Render mode | cpu — Mesa lavapipe 26.2.1 |
| Episodes | RoboMME test split, 50 per task, 4 shards, one model server per shard |
| Slurm jobs | 6798706 Permanence (1:34:12), 6798707 Counting/PickXtimes (0:16:13) — 1.9 GPU-hours |

| Task | Ours: SR (95% Wilson CI) | Published | Match |
|---|---|---|---|
| ButtonUnmask | 16.0% [8.3, 28.5] | — | |
| ButtonUnmaskSwap | 8.0% [3.2, 18.8] | — | |
| VideoUnmask | 26.0% [15.9, 39.6] | — | |
| VideoUnmaskSwap | 20.0% [11.2, 33.0] | — | |
| **Permanence (suite)** | **17.5% [12.9, 23.4]** | 13.67% | ✅ inside CI |
| PickXtimes | 34.0% [22.4, 47.8] | 36% (harness's x86 run) | ✅ inside CI |

No episode errored. This doubles as the aarch64 + CPU-rendering parity check: both reference
numbers fall inside our intervals, so the platform is not obviously shifting results.

Caveats before quoting these:

- Published numbers average 9 runs (3 checkpoints × 3 seeds). Hugging Face ships one checkpoint
  per variant, so ours is 1 checkpoint × 1 seed and carries more variance than the interval
  alone suggests.
- Per-task Permanence numbers from the RoboMME paper are not in the harness's report; fill the
  column in from the paper's per-task tables.
- 50 episodes give a CI half-width of roughly ±14 points at 50% success — fine for reproduction,
  too wide for the paper's comparisons (§2.5).

### Free first look at H1 (§2.6)

For the memoryless baseline, the Swap variants are already harder: ButtonUnmask 16.0% → Swap
8.0% (Δ 8.0 pp) and VideoUnmask 26.0% → Swap 20.0% (Δ 6.0 pp). A model with no memory cannot be
losing memory of the swap, so this is task difficulty, not an agent-causality effect. It sets a
floor: a memory variant's Swap gap only means something if it exceeds this.

## Pending — memory variants

FrameSamp+Modulator and TTT-Expert were submitted (jobs 6798708–6798711) and stopped: they
cannot be reproduced through the harness as shipped. The harness pushes the conditioning video
into memory once, with zero-filled proprioception, and never updates it; MME-VLA's own eval
client seeds memory from the episode's initial frames and streams observations with real states
before every inference. On Counting tasks, which have no conditioning video, the harness path
errors immediately with `history feats is empty, add buffer first`.

The protocol to use is an open decision, not a bug to patch blindly: what enters memory and when
is part of the method under study. See `scripts/isambard/README.md` for the reading list.
