# Memory probes: does VLA memory track *when*, or only *what*?

## Question

MME-VLA's memory variants beat the memoryless π0.5 on RoboMME's memory tasks. Do they remember
what happened, in order, or do they pool visual features from the frames they keep? If memory
were a causal record, destroying the temporal order of its input should hurt tasks that need
order and spare tasks that do not. If it is a bag of features, order should not matter anywhere
while removing the content should.

## Setting

- **Model:** FrameSamp+Modulator (perceptual memory, modulation integration), the paper's best
  variant (44.5% average).
- **Memory input:** under the harness, memory holds the conditioning video, pushed once at episode
  start. FrameSamp keeps 32 evenly spaced frames of it (512-token budget, 16 tokens per frame) with
  positional embeddings, fixed for the whole episode; proprioception is unused
  (`use_state_emb: false`). MME-VLA's own eval client additionally streams the robot's
  observations into memory, so these results compare conditions *within* this protocol and are
  not the paper's numbers.
- **Not probed:** TTT-Expert. The recurrent variants cannot run through the harness at all — every
  episode fails an assertion on `exec_start_idx` — so they wait on the memory-protocol decision.
- **Tasks:** 50 fixed-seed test episodes each (26 easy, 12 medium, 12 hard):
  - `VideoUnmaskSwap` — a shell game. Cubes go under 3–4 bins, then the bins swap 1–3 times.
    Picking the right bin requires following the swaps in order.
  - `VideoUnmask` — the same scene without swaps. What is under each bin is visible before it is
    covered, so content matters and order does not. This is the control.
- **Floor:** the memoryless π0.5 baseline on the same episodes (`pi05_baseline_permanence_6798706`).

## Conditions

Only the frames the model receives as memory change; scene, seed and task are identical, so
every comparison is paired episode by episode.

| Condition | Memory input | Isolates |
|---|---|---|
| `none` | original video | reference |
| `none_rep` | original video, rerun | run-to-run noise |
| `blank` | black frames, same length | whether memory content is used at all |
| `shuffle` | all frames permuted | order, plus which 32 frames get kept |
| `shuffle_kept` | only the 32 kept frames permuted | **order alone**: identical content |
| `reverse` | frames reversed | causal direction (same kept frames, ±1) |

Permutations are seeded by (task, episode), so they are identical across reruns and models, and
each run logs the source frame shown in every memory slot.

## Predictions

| If memory… | `VideoUnmask` | `VideoUnmaskSwap` |
|---|---|---|
| tracks events in order | `blank` hurts; order conditions ≈ `none` | `blank`, `shuffle_kept` and `reverse` all hurt |
| pools frame features | `blank` hurts; order conditions ≈ `none` | `blank` hurts; order conditions ≈ `none` |
| is not used | everything ≈ `none` ≈ π0.5 | everything ≈ `none` ≈ π0.5 |

`VideoUnmaskSwap` separates the first two readings; `VideoUnmask` checks that a perturbation
does not simply break the model.

## Running

```bash
cd ~/code/MemVLA && source scripts/isambard/env.sh
experiments/memory_probes/submit.sh                  # 12 jobs, ~6 GPU-hours
$MEMVLA_SIM_ENV/bin/python experiments/memory_probes/analyze.py $MEMVLA_ROOT/results \
    --baseline $MEMVLA_ROOT/results/pi05_baseline_permanence_6798706
```

## Results (23 Sep 2026, jobs 6833867–6833878)

600 episodes, none errored; every perturbed episode logged its perturbation. 6.4 GPU-hours.
Δ is the paired change against the clean run on the same episodes; "lost / gained" counts the
episodes that flipped; p is an exact McNemar test.

| Condition | `VideoUnmask` | Δ (lost / gained, p) | `VideoUnmaskSwap` | Δ (lost / gained, p) |
|---|---|---|---|---|
| `none` | 30% (15/50) | — | 10% (5/50) | — |
| `none_rep` | 30% | ±0 (0 / 0) | 10% | ±0 (0 / 0) |
| `blank` | 18% | −12 pp (7 / 1, p = 0.07) | 8% | −2 pp (3 / 2, p = 1.0) |
| `shuffle` | 26% | −4 pp (4 / 2, p = 0.69) | 10% | ±0 (2 / 2, p = 1.0) |
| `shuffle_kept` | 22% | −8 pp (7 / 3, p = 0.34) | 8% | −2 pp (3 / 2, p = 1.0) |
| `reverse` | 20% | −10 pp (6 / 1, p = 0.13) | 12% | +2 pp (1 / 2, p = 1.0) |
| π0.5, no memory | 26% | −4 pp (13 / 11, p = 0.84) | 20% | +10 pp (4 / 9, p = 0.27) |

### What the data show

1. **The pipeline is deterministic.** The clean rerun matches episode for episode on both tasks,
   so every flipped episode under a perturbation is caused by the perturbation, not noise.
2. **Memory input is live.** On `VideoUnmask`, blanking memory loses 7 episodes and gains 1
   (−12 pp, p = 0.07). The model does read the conditioning video.
3. **On the order-critical task there is no memory benefit to remove.** Clean FrameSamp solves
   5 of 50 `VideoUnmaskSwap` episodes, fewer than the memoryless π0.5 (10 of 50). A perturbation
   can only destroy what memory contributes, and here it contributes nothing measurable. So the
   flat results on Swap are a floor effect: they say nothing about whether the model uses order.
   **They are not evidence for the bag-of-features hypothesis.**
4. **On `VideoUnmask`, order perturbations lower success, none significantly.** Reversal
   (−10 pp) costs nearly as much as blanking (−12 pp), although it preserves every fact in the
   video. That hints the model reads memory by position — for instance, taking the latest frames
   as the current state — rather than by what the frames mean. With 7 or fewer flipped episodes
   per comparison, this is suggestive only.
5. **Under this protocol, the paper's memory advantage does not appear.** FrameSamp scores 30%
   vs π0.5's 26% on `VideoUnmask`, and below it on Swap. The paper reports 36.0% vs 13.7% on the
   Permanence suite. The memory-feeding protocol (video once, versus MME-VLA's streaming) is the
   prime suspect.

### Verdict

The hypothesis is neither supported nor refuted. The task built to test it sits at floor for this
model under this protocol, and the control task is too underpowered to separate order from
content.

### What would make it decisive

- **Settle the memory protocol first**, then rerun `none` on `VideoUnmaskSwap`. The probe is only
  meaningful once clean memory beats no memory on an order-critical task.
- **Measure the choice, not just success.** Log which bin the robot lifts on Swap. If it goes to
  where the target cube started, before the swaps, memory holds content without dynamics: a direct
  test of the hypothesis that does not depend on the success rate.
- **More power.** The pipeline is deterministic, so paired designs are cheap, but 50 test
  episodes cap each comparison at a handful of flips. The `val` and `train` splits in the repo
  hold only 3 episodes per task, so more seeds need a custom episode file: RoboMME's env builder
  accepts `override_metadata_path`, which the probe benchmark would have to pass through. Use such
  seeds for probing only, never for reported test numbers.
