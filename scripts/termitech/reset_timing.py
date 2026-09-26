"""Reset one RoboDojo task through layouts 0..COUNT-1 in a single process, as the harness does
before every episode, and report per reset: whether the layout settled, how long each phase took,
how many prims the stage holds and which subtrees grew. A reset should take about as long at the
last layout as at the second; growth means something leaks (see README, "Fixes to RoboDojo").

    source scripts/termitech/env.sh
    CUDA_VISIBLE_DEVICES=1 micromamba run -p $ROBODOJO_ENV python scripts/termitech/reset_timing.py press_by_number 12

No policy or model server is involved. Isaac Sim is killed with os._exit at the end, since it
hangs or crashes on a normal shutdown.
"""

import argparse
from collections import Counter, defaultdict
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("count", type=int)
parser.add_argument("--group", type=int, default=0, help="Eval_Layout group (the harness's seed param)")
args = parser.parse_args()

from vla_eval.benchmarks.robodojo.benchmark import RoboDojoBenchmark  # noqa: E402

bench = RoboDojoBenchmark(root=os.environ["ROBODOJO_ROOT"], tasks=[args.task], seed=args.group)
env = bench._build_env(args.task)

from isaacsim.core.utils.stage import get_current_stage  # noqa: E402  (importable once Kit runs)

totals, calls = defaultdict(float), defaultdict(int)


def timed(owner, name, label):
    fn = getattr(owner, name)

    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            totals[label] += time.perf_counter() - t0
            calls[label] += 1

    setattr(owner, name, wrapper)


sm = env.scene_manager
for owner, name in [
    (env, "sim_step"),
    (env, "render"),
    (sm, "reload_scene"),
    (sm, "apply_saved_poses"),
    (sm.layout_manager, "check_layout_stability"),
    (env.obs_manager, "get_obs"),
    (env.capture_manager, "reset"),
]:
    timed(owner, name, f"{type(owner).__name__}.{name}")

stage = get_current_stage()
previous = None
for layout_id in range(args.count):
    totals.clear()
    calls.clear()
    t0 = time.perf_counter()
    try:
        env.reset(seed=[layout_id])
        outcome = "settled"
    except Exception as e:  # noqa: BLE001
        outcome = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0
    paths = {str(p.GetPath()) for p in stage.Traverse()}
    print(f"layout {layout_id}: {outcome}; reset {elapsed:.1f}s; {len(paths)} prims", flush=True)
    for label in sorted(totals, key=lambda k: -totals[k]):
        print(f"    {label:45s} {totals[label]:7.2f}s in {calls[label]:4d} calls "
              f"({1000 * totals[label] / calls[label]:.1f} ms each)", flush=True)
    if previous is not None:
        subtree = lambda p: "/".join(p.split("/")[:6])  # noqa: E731
        added, removed = Counter(map(subtree, paths - previous)), Counter(map(subtree, previous - paths))
        for key, n in sorted(((k, added[k] - removed[k]) for k in added), key=lambda x: -x[1])[:5]:
            if n > 0:
                print(f"    grew by {n:4d} prims: {key}", flush=True)
    previous = paths
os._exit(0)
