#!/bin/bash
# RoboDojo assets (~90 GB) and the released π0.5 checkpoint (training seed 0, ~7 GB) from the
# RoboDojo-Benchmark/RoboDojo dataset (Apache-2.0, ungated), through the Hugging Face mirror.
# Same files as upstream's scripts/init_assets.sh, which needs git-lfs and huggingface.co.
#
# Why not `hf download --include`: the mirror's paginated file listing returns next-page links to
# huggingface.co, which is blocked here, so it retries forever. The dataset-info call lists all
# files in one response, and per-file downloads go through the mirror. Xet transfers are disabled
# for the same reason (their endpoint is blocked too). Resumable: re-run after an interruption.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export HF_HUB_DISABLE_XET=1 REVISION=43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab   # dataset commit used

uvx --quiet --from huggingface_hub python - <<'EOF'
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

repo, rev = "RoboDojo-Benchmark/RoboDojo", os.environ["REVISION"]
dest = os.path.join(os.environ["MEMVLA_DATA"], "robodojo")
prefixes = ("Assets/", "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/")
files = [s.rfilename for s in HfApi().dataset_info(repo, revision=rev).siblings if s.rfilename.startswith(prefixes)]
print(f"{len(files)} files to fetch or verify at {rev[:7]}", flush=True)

def fetch(name):
    for attempt in range(5):
        try:
            return hf_hub_download(repo, name, repo_type="dataset", revision=rev, local_dir=dest)
        except Exception as exc:          # the mirror drops the odd connection; back off and retry
            if attempt == 4:
                raise RuntimeError(f"{name}: {exc}") from exc
            time.sleep(2 ** attempt)

start, done = time.time(), 0
with ThreadPoolExecutor(max_workers=16) as pool:
    for future in as_completed([pool.submit(fetch, f) for f in files]):
        future.result()
        done += 1
        if done % 500 == 0 or done == len(files):
            print(f"{done}/{len(files)} files, {time.time() - start:.0f}s", flush=True)
EOF

for sub in Robots Object Material Eval_Layout; do     # init_assets.sh's completeness check
    [ -d "$ROBODOJO_ASSETS/$sub" ] || { echo "missing $ROBODOJO_ASSETS/$sub" >&2; exit 1; }
done
[ -d "$ROBODOJO_PI05_CKPT/params" ] || { echo "missing $ROBODOJO_PI05_CKPT/params" >&2; exit 1; }
ln -sfn "$ROBODOJO_ASSETS" "$ROBODOJO_ROOT/Assets"    # where RoboDojo's own code looks
du -sh --apparent-size "$ROBODOJO_ASSETS" "$(dirname "$ROBODOJO_PI05_CKPT")"
