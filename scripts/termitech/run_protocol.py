"""Evaluate π0.5 on RoboDojo tasks under the published protocol, spread over several GPUs.

Each task's layouts are split into chunks (10 by default) that run as separate simulator
processes, one per GPU, longest tasks first. The simulators share π0.5 servers, which sit on
the same GPUs with a capped memory share: π0.5 infers once per 50-step action chunk, so one
server keeps up with many simulators. A chunk whose simulator dies is re-run from its next
layout. Afterwards each task's result follows the protocol: its first N episodes in layout
order, where N is the task's episode count in the harness's eval.yaml; layouts that fail to
build are skipped, as the harness does, and made up from the next ones.

    source scripts/termitech/env.sh
    PY=$ROBODOJO_ENV/bin/python
    setsid nohup $PY scripts/termitech/run_protocol.py run --tasks cover_blocks,swap_T \\
        --gpus 1,3,4,5,6,7 --server-gpus 1,5 > $MEMVLA_DATA/logs/protocol.log 2>&1 < /dev/null &
    $PY scripts/termitech/run_protocol.py summarize $MEMVLA_DATA/results/protocol_<time>

`run` writes everything to $MEMVLA_DATA/results/protocol_<time>/ (state.json shows progress)
and stops its servers and simulators when it ends or gets SIGTERM.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request

import yaml

HARNESS = Path(os.environ["MEMVLA_CODE"]) / "vla-evaluation-harness"
ROBODOJO_ROOT = Path(os.environ["ROBODOJO_ROOT"])
ROBODOJO_ENV = os.environ["ROBODOJO_ENV"]
ASSETS = Path(os.environ["ROBODOJO_ASSETS"])
EVAL_YAML = HARNESS / "configs/benchmarks/robodojo/eval.yaml"
SERVER_YAML = "configs/model_servers/robodojo_pi05/pi05.yaml"
RESULT_FILE = "RoboDojoBenchmark_aggregate.json"
SERVER_MEM_FRACTION = "0.35"  # of an 80 GB A100; the simulator on the same GPU needs ~10-20 GB
POLL = 15  # seconds
EXIT_GRACE = 60  # seconds a simulator gets to exit after writing results (Isaac Sim hangs on shutdown)
MAX_TRIES = 3  # per chunk


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def protocol_config(task):
    """The harness's eval.yaml reduced to this task's entry, and the task's episode count."""
    cfg = yaml.safe_load(EVAL_YAML.read_text())
    for entry in cfg["benchmarks"]:
        if task in entry.get("params", {}).get("tasks", []):
            one = copy.deepcopy(entry)
            one["params"]["tasks"] = [task]
            cfg["benchmarks"] = [one]
            return cfg, int(one.get("episodes_per_task", 50))
    sys.exit(f"{task} is not a task in {EVAL_YAML}")


def step_limit(task):
    text = (ROBODOJO_ROOT / "task/RoboDojo/tasks" / f"{task}.py").read_text()
    match = re.search(r"self\.step_lim\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else 1000


def layout_count(task, group):
    pattern = re.compile(rf"{re.escape(task)}_\d+\.json")
    folder = ASSETS / f"Eval_Layout/RoboDojo/arx_x5/{group}"
    return sum(1 for f in folder.iterdir() if pattern.fullmatch(f.name))


def recorded_episodes(folder):
    """Episodes a chunk recorded: one dict per episode that ran (errors excluded)."""
    rows = []
    for db_path in Path(folder).glob("recording-*.sqlite"):
        for attempt in range(5):
            try:
                db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
                fetched = db.execute("select status, metrics, steps, elapsed_sec from episode_results").fetchall()
                db.close()
                break
            except sqlite3.Error:
                time.sleep(2)
        else:
            continue
        for status, metrics, steps, elapsed in fetched:
            m = json.loads(metrics or "{}")
            layout = m.get("layout_id")
            if status in ("success", "fail") and isinstance(layout, int) and layout >= 0:
                rows.append({"layout": layout, "success": bool(m.get("success")),
                             "score": float(m.get("score") or 0.0), "steps": int(steps or 0),
                             "seconds": float(elapsed or 0.0)})
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


# --------------------------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------------------------

def summarize(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    lines, results = [], {}
    header = f"{'task':30s} {'episodes':>8s} {'success':>8s} {'95% CI':>13s} {'score':>6s}  layouts        steps/min"
    lines += [header, "-" * len(header)]
    for task, target in meta["targets"].items():
        by_layout = {}
        for folder in sorted((run_dir / task).glob("layouts_*")):
            for row in recorded_episodes(folder):
                by_layout.setdefault(row["layout"], row)
        counted = [by_layout[k] for k in sorted(by_layout)][:target]
        n = len(counted)
        wins = sum(r["success"] for r in counted)
        score = 100 * sum(r["score"] for r in counted) / n if n else 0.0
        low, high = wilson(wins, n)
        rate = sum(r["steps"] for r in counted) / max(1.0, sum(r["seconds"] for r in counted) / 60)
        span = f"{counted[0]['layout']}-{counted[-1]['layout']}" if counted else "-"
        results[task] = {"episodes": n, "target": target, "successes": wins,
                         "success_rate": 100 * wins / n if n else 0.0, "ci95": [low, high], "score": score,
                         "layouts": [r["layout"] for r in counted]}
        note = "" if n >= target else f"  ({target - n} missing)"
        lines.append(f"{task:30s} {n:>4d}/{target:<3d} {results[task]['success_rate']:7.1f}% "
                     f"[{low:4.1f},{high:5.1f}] {score:6.1f}  {span:13s} {rate:6.0f}{note}")
    done = [r for r in results.values() if r["episodes"]]
    if done:
        mean_sr = sum(r["success_rate"] for r in done) / len(done)
        mean_score = sum(r["score"] for r in done) / len(done)
        lines.append("-" * len(header))
        lines.append(f"{'mean over ' + str(len(done)) + ' tasks':30s} {'':8s} {mean_sr:7.1f}% {'':13s} {mean_score:6.1f}")
    text = "\n".join(lines)
    (run_dir / "summary.txt").write_text(text + "\n")
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(text, flush=True)
    return results


# --------------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------------

class Server:
    def __init__(self, gpu, run_dir):
        self.gpu, self.port = gpu, 18200 + gpu
        self.log = run_dir / "logs" / f"server_gpu{gpu}.log"
        self.proc = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def healthy(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5):
                return True
        except OSError:
            return False

    def start(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu), XLA_PYTHON_CLIENT_MEM_FRACTION=SERVER_MEM_FRACTION)
        with open(self.log, "a") as out:
            self.proc = subprocess.Popen(
                ["uv", "run", "vla-eval", "serve", "-c", SERVER_YAML, "--address", f"127.0.0.1:{self.port}"],
                cwd=HARNESS, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.healthy():
                log(f"π0.5 server up on GPU {self.gpu} (port {self.port})")
                return
            if not self.alive():
                break
            time.sleep(10)
        self.stop()
        sys.exit(f"π0.5 server on GPU {self.gpu} did not start; see {self.log}")

    def stop(self):
        if self.proc is not None:
            kill_group(self.proc)
            self.proc = None


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


class Chunk:
    def __init__(self, task, start, stop, tries=0):
        self.task, self.start, self.stop, self.tries = task, start, stop, tries
        self.state, self.gpu, self.proc, self.folder = "queued", None, None, None
        self.started = self.finished = self.result_seen = None

    def as_dict(self):
        return {k: v for k, v in vars(self).items() if k != "proc" and not isinstance(v, Path)} | {
            "folder": str(self.folder) if self.folder else None}


def run(args):
    tasks = [t for t in args.tasks.split(",") if t]
    gpus = [int(g) for g in args.gpus.split(",")]
    server_gpus = [int(g) for g in args.server_gpus.split(",")]
    run_dir = Path(os.environ["MEMVLA_DATA"]) / "results" / f"protocol_{args.name or time.strftime('%Y%m%d_%H%M%S')}"
    for sub in ("configs", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    targets, groups, limits, sizes = {}, {}, {}, {}
    for task in tasks:
        cfg, episodes = protocol_config(task)
        targets[task] = args.episodes or episodes
        groups[task] = int(cfg["benchmarks"][0]["params"].get("seed", 0))
        limits[task] = step_limit(task)
        sizes[task] = layout_count(task, groups[task])
        (run_dir / "configs" / f"{task}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    commits = {name: subprocess.run(["git", "-C", str(Path(os.environ["MEMVLA_CODE"]) / name), "rev-parse", "--short",
                                     "HEAD"], capture_output=True, text=True).stdout.strip()
               for name in ("RoboDojo", "vla-evaluation-harness")}
    (run_dir / "run.json").write_text(json.dumps({
        "tasks": tasks, "targets": targets, "layout_groups": groups, "step_limits": limits, "layouts": sizes,
        "gpus": gpus, "server_gpus": server_gpus, "chunk": args.chunk, "max_steps": args.max_steps,
        "checkpoint": os.environ.get("ROBODOJO_PI05_CKPT"), "commits": commits,
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    log(f"run folder {run_dir}")

    queue = []
    for task in tasks:
        for start in range(0, min(targets[task], sizes[task]), args.chunk):
            queue.append(Chunk(task, start, min(start + args.chunk, targets[task], sizes[task])))
    queue.sort(key=lambda c: -limits[c.task] * (c.stop - c.start))  # longest first
    chunks = list(queue)

    servers = [Server(g, run_dir) for g in server_gpus]
    lanes = {gpu: None for gpu in gpus}  # gpu -> running chunk

    def shutdown(*_):
        log("stopping: killing simulators and servers")
        for chunk in lanes.values():
            if chunk is not None and chunk.proc is not None:
                kill_group(chunk.proc)
        for server in servers:
            server.stop()
        save_state()
        sys.exit(1)

    def save_state():
        (run_dir / "state.json").write_text(json.dumps([c.as_dict() for c in chunks], indent=1) + "\n")

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    for server in servers:
        server.start()

    def launch(chunk, gpu):
        server = servers[gpus.index(gpu) % len(servers)]
        if not server.alive() or not server.healthy():
            log(f"π0.5 server on GPU {server.gpu} is down; restarting it")
            server.stop()
            server.start()
        chunk.tries += 1
        chunk.folder = run_dir / chunk.task / f"layouts_{chunk.start:02d}-{chunk.stop - 1:02d}_try{chunk.tries}"
        cmd = ["micromamba", "run", "-p", ROBODOJO_ENV, "vla-eval", "run", "--no-docker",
               "-c", str(run_dir / "configs" / f"{chunk.task}.yaml"),
               "--param", f"root={ROBODOJO_ROOT}",
               "--param", f"layout_start={chunk.start}", "--param", f"layout_stop={chunk.stop}",
               "--benchmark-field", f"episodes_per_task={chunk.stop - chunk.start}",
               "--server-url", f"ws://127.0.0.1:{server.port}", "--output-dir", str(chunk.folder), "--record-video"]
        if args.max_steps:
            cmd += ["--benchmark-field", f"max_steps={args.max_steps}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log_path = run_dir / "logs" / f"{chunk.folder.parent.name}_{chunk.folder.name}.log"
        with open(log_path, "w") as out:
            chunk.proc = subprocess.Popen(cmd, cwd=HARNESS, env=env, stdout=out, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
        chunk.state, chunk.gpu, chunk.started, chunk.result_seen = "running", gpu, time.time(), None
        lanes[gpu] = chunk
        log(f"start {chunk.task} layouts {chunk.start}-{chunk.stop - 1} on GPU {gpu} (try {chunk.tries})")

    def finish(chunk, state):
        kill_group(chunk.proc)
        chunk.proc, chunk.state, chunk.finished = None, state, time.time()
        lanes[chunk.gpu] = None
        rows = recorded_episodes(chunk.folder)
        minutes = (chunk.finished - chunk.started) / 60
        log(f"{state} {chunk.task} layouts {chunk.start}-{chunk.stop - 1} on GPU {chunk.gpu}: {len(rows)} episodes, "
            f"{sum(r['success'] for r in rows)} successes, {minutes:.0f} min")
        return rows

    def top_ups():
        """Chunks for tasks that have fewer episodes than their target, from the next unused layouts."""
        new = []
        for task in tasks:
            mine = [c for c in chunks if c.task == task]
            if any(c.state in ("queued", "running") for c in mine):
                continue
            counted = {r["layout"] for c in mine if c.folder for r in recorded_episodes(c.folder)}
            missing = targets[task] - len(counted)
            next_layout = max(c.stop for c in mine)
            if missing > 0 and next_layout < sizes[task]:
                stop = min(next_layout + missing, sizes[task])
                log(f"{task}: {missing} episodes short; adding layouts {next_layout}-{stop - 1}")
                new.append(Chunk(task, next_layout, stop))
            elif missing > 0:
                log(f"{task}: {missing} episodes short and no layouts left")
        return new

    while True:
        for gpu, chunk in lanes.items():
            if chunk is None and queue:
                launch(queue.pop(0), gpu)
        save_state()
        time.sleep(POLL)
        for gpu, chunk in list(lanes.items()):
            if chunk is None:
                continue
            if (chunk.folder / RESULT_FILE).exists():
                chunk.result_seen = chunk.result_seen or time.time()
                if chunk.proc.poll() is not None or time.time() - chunk.result_seen > EXIT_GRACE:
                    finish(chunk, "done")
            elif chunk.proc.poll() is not None:
                rows = finish(chunk, "crashed")
                resume = max((r["layout"] for r in rows), default=chunk.start - 1) + 1
                if resume < chunk.stop and chunk.tries < MAX_TRIES:
                    retry = Chunk(chunk.task, resume, chunk.stop, tries=chunk.tries)
                    chunks.append(retry)
                    queue.insert(0, retry)
                    log(f"re-queued {chunk.task} layouts {resume}-{chunk.stop - 1}")
        if not queue and all(c is None for c in lanes.values()):
            extra = top_ups()
            if not extra:
                break
            chunks.extend(extra)
            queue.extend(extra)
    save_state()
    for server in servers:
        server.stop()
    log("all chunks finished")
    summarize(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="evaluate tasks")
    r.add_argument("--tasks", required=True, help="comma-separated task names")
    r.add_argument("--gpus", default="1,3,4,5,6,7", help="GPUs for simulators, one simulator each")
    r.add_argument("--server-gpus", default="1,5", help="GPUs that also host a π0.5 server")
    r.add_argument("--chunk", type=int, default=10, help="layouts per simulator process")
    r.add_argument("--episodes", type=int, help="episodes per task instead of the protocol's (for tests)")
    r.add_argument("--max-steps", type=int, help="cap steps per episode (for tests)")
    r.add_argument("--name", help="run folder suffix (default: the start time)")
    s = sub.add_parser("summarize", help="combine a run's chunks into per-task results")
    s.add_argument("run_dir")
    args = parser.parse_args()
    run(args) if args.command == "run" else summarize(args.run_dir)


if __name__ == "__main__":
    main()
