"""Summarise harness aggregate JSONs: success rate per task with 95% Wilson intervals.

Wilson rather than normal intervals because success rates here sit near 0 and the episode
counts are small. Episodes that never stepped are reported separately: they are harness or
server errors, not task failures, and they drag the rate down silently.

Usage:
    python scripts/analyze_results.py $MEMVLA_ROOT/results/pi05_baseline_*/
"""

import json
import math
import sys
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def summarise(aggregate: Path) -> None:
    data = json.loads(aggregate.read_text())
    print(f"\n{aggregate.parent.name}  [{data['benchmark']}]")
    total_ok = total_n = total_err = 0
    for task in data["tasks"]:
        episodes = task["episodes"]
        n = len(episodes)
        ok = sum(bool(e.get("metrics", {}).get("success")) for e in episodes)
        errored = sum(1 for e in episodes if not e.get("steps"))
        lo, hi = wilson(ok, n)
        note = f"  ({errored} errored)" if errored else ""
        print(f"  {task['task']:<18} {100 * ok / n:5.1f}%  {ok:3d}/{n:<3d}  95% CI [{lo:4.1f}, {hi:4.1f}]{note}")
        total_ok, total_n, total_err = total_ok + ok, total_n + n, total_err + errored
    if len(data["tasks"]) > 1:
        lo, hi = wilson(total_ok, total_n)
        print(f"  {'SUITE':<18} {100 * total_ok / total_n:5.1f}%  {total_ok:3d}/{total_n:<3d}  95% CI [{lo:4.1f}, {hi:4.1f}]")
    if total_err:
        print(f"  note: {total_err} episodes never stepped — check the shard logs before quoting these numbers")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sys.exit(__doc__)
    for path in paths:
        for aggregate in sorted(path.glob("*_aggregate.json") if path.is_dir() else [path]):
            summarise(aggregate)
