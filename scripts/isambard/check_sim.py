"""Simulator sanity check for the RoboMME env on Isambard-AI (no model needed).

Engages lavapipe CPU rendering, resets one test episode (which runs mplib motion planning to
render the conditioning video), then holds the arm still for a few steps.

Run it with the GPU hidden, as the harness does for CPU rendering:
    CUDA_VISIBLE_DEVICES= python check_sim.py [TASK] [EPISODE]
"""

import sys
import time

import numpy as np
from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark
from vla_eval.recording import NullEpisodeRecorder

task_name = sys.argv[1] if len(sys.argv) > 1 else "PickXtimes"
episode = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print("render env:", RoboMMEBenchmark.configure_render("cpu"), flush=True)
bench = RoboMMEBenchmark(tasks=[task_name])
bench._recorder = NullEpisodeRecorder()  # the harness attaches this in start_episode()
task = {"name": task_name, "env_id": task_name, "episode_idx": episode}

t0 = time.time()
obs = bench.make_obs(bench.reset(task), task)
images = {k: v.shape for k, v in obs["images"].items()}
print(
    f"reset {task_name} ep{episode}: {time.time() - t0:.1f}s | task: {obs['task_description']!r} | "
    f"images {images} | conditioning video: {len(obs.get('video_history', []))} frames",
    flush=True,
)

hold = np.asarray(obs["states"][:8], dtype=np.float32)  # 7 joint angles + gripper
t0 = time.time()
for _ in range(20):
    bench.step({"actions": hold})
print(f"20 steps: {(time.time() - t0) / 20 * 1000:.0f} ms/step")
bench.cleanup()
