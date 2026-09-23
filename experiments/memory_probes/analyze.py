"""Compare memory-perturbation conditions episode by episode.

Reads probe results from folders named ``<model>_permanence_<Task>_<condition>_<jobid>``, pairs
every condition with the clean run on the same test episodes, and reports:
    - success rate with a 95% Wilson interval,
    - the paired change against clean, with a 95% bootstrap interval over episodes,
    - an exact McNemar test on the discordant episodes (clean-only vs condition-only successes).
A memoryless baseline run (--baseline) is paired against clean the same way, as the floor.

Usage:
    python experiments/memory_probes/analyze.py $MEMVLA_ROOT/results \
        --baseline $MEMVLA_ROOT/results/pi05_baseline_permanence_6798706
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ORDER = ["none", "none_rep", "blank", "shuffle", "shuffle_kept", "reverse"]
RUN = re.compile(r"^(?P<model>.+)_permanence_(?P<task>Video\w+?)_(?P<cond>[a-z_]+)_(?P<job>\d+)$")


def outcomes(aggregate: Path, task: str) -> dict[int, bool]:
    """episode_idx -> success, skipping episodes that never stepped (harness or server errors)."""
    data = json.loads(aggregate.read_text())
    result = {}
    for entry in data["tasks"]:
        if entry["task"] != task:
            continue
        for ep in entry["episodes"]:
            if ep.get("steps"):
                result[ep["episode_idx"]] = bool(ep.get("metrics", {}).get("success"))
    return result


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(ref: dict[int, bool], cond: dict[int, bool], rng: np.random.Generator):
    eps = sorted(set(ref) & set(cond))
    r = np.array([ref[e] for e in eps], dtype=float)
    x = np.array([cond[e] for e in eps], dtype=float)
    b = int(((r == 1) & (x == 0)).sum())  # lost under the condition
    c = int(((r == 0) & (x == 1)).sum())  # gained under the condition
    diffs = x - r
    boot = rng.choice(diffs, size=(10_000, len(diffs)), replace=True).mean(axis=1) * 100
    return len(eps), 100 * diffs.mean(), np.percentile(boot, [2.5, 97.5]), b, c, mcnemar_exact(b, c)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", type=Path, help="memoryless run on the same tasks and episodes")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    # Keep the largest run per (model, task, condition), so smoke tests never shadow the grid
    runs: dict[tuple[str, str, str], tuple[int, Path]] = {}
    for folder in args.results.iterdir():
        m = RUN.match(folder.name)
        aggregates = list(folder.glob("*_aggregate.json"))
        if not m or not aggregates:
            continue
        key = (m["model"], m["task"], m["cond"])
        n = len(outcomes(aggregates[0], m["task"]))
        if key not in runs or n > runs[key][0]:
            runs[key] = (n, aggregates[0])

    for model, task in sorted({(k[0], k[1]) for k in runs}):
        print(f"\n### {model} — {task}\n")
        print("| Condition | Success (95% CI) | Δ vs clean (95% CI) | Lost / gained | McNemar p |")
        print("|---|---|---|---|---|")
        ref = outcomes(runs[(model, task, "none")][1], task) if (model, task, "none") in runs else None
        rows = [(c, outcomes(runs[(model, task, c)][1], task)) for c in ORDER if (model, task, c) in runs]
        if args.baseline:
            for agg in args.baseline.glob("*_aggregate.json"):
                rows.append(("no memory (π0.5)", outcomes(agg, task)))
        for cond, res in rows:
            k, n = sum(res.values()), len(res)
            lo, hi = wilson(k, n)
            cell = f"{100 * k / n:.0f}% ({k}/{n}) [{lo:.0f}, {hi:.0f}]"
            if ref is None or cond == "none":
                print(f"| {cond} | {cell} | — | — | — |")
                continue
            _, delta, (dlo, dhi), b, c, p = paired(ref, res, rng)
            print(f"| {cond} | {cell} | {delta:+.0f} pp [{dlo:+.0f}, {dhi:+.0f}] | {b} / {c} | {p:.3f} |")


if __name__ == "__main__":
    main()
