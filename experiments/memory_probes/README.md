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

## Results

Pending: jobs 6833867–6833878.
