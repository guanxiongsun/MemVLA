#!/bin/bash
# RoboDojo assets (41 GB) and the released π0.5 checkpoint (training seed 0, 45 GB: 12 GB of params
# and 32 GB of optimizer state), 86 GB in all, pinned to the Hugging Face dataset revision below and
# verified file by file against it (size and SHA-256).
# Same files as upstream's scripts/init_assets.sh, which needs git-lfs and huggingface.co.
#
# Transport: huggingface.co is blocked from this machine, and its mirror serves ~2.5 MB/s. ModelScope
# hosts the same dataset (Apache-2.0) and is faster here, on many connections at once, so files come
# from ModelScope and are accepted only if they match the pinned revision's hash; anything that
# fails falls back to the mirror. Files already present at the right size are kept. Resumable.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export REVISION=43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab    # dataset commit used

uvx --quiet --from huggingface_hub --with requests python - <<'EOF'
import hashlib, os, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from huggingface_hub import HfApi

repo, rev = "RoboDojo-Benchmark/RoboDojo", os.environ["REVISION"]
dest = os.path.join(os.environ["MEMVLA_DATA"], "robodojo")
prefixes = ("Assets/", "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/")
# One listing call (the mirror's paginated tree listing links back to blocked huggingface.co)
files = [s for s in HfApi().dataset_info(repo, revision=rev, files_metadata=True).siblings
         if s.rfilename.startswith(prefixes)]
modelscope = "https://modelscope.cn/datasets/" + repo + "/resolve/master/{}"
mirror = os.environ["HF_ENDPOINT"] + f"/datasets/{repo}/resolve/{rev}/{{}}"
print(f"{len(files)} files, {sum(f.size for f in files) / 1e9:.1f} GB at revision {rev[:7]}", flush=True)

def fetch(f):
    path = os.path.join(dest, f.rfilename)
    if os.path.exists(path) and os.path.getsize(path) == f.size:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp, error = path + ".part", None
    for source in [modelscope] * 3 + [mirror] * 2:
        try:
            sha = hashlib.sha256()
            with requests.get(source.format(urllib.parse.quote(f.rfilename)), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as out:
                    for chunk in r.iter_content(4 << 20):
                        out.write(chunk)
                        sha.update(chunk)
            if os.path.getsize(tmp) != f.size or (f.lfs and sha.hexdigest() != f.lfs.sha256):
                raise ValueError("size or SHA-256 differs from the pinned revision")
            os.replace(tmp, path)
            return f.size
        except Exception as exc:
            error = exc
            time.sleep(5)
    raise RuntimeError(f"{f.rfilename}: {error}")

start, done, fetched = time.time(), 0, 0
with ThreadPoolExecutor(max_workers=16) as pool:
    for future in as_completed([pool.submit(fetch, f) for f in files]):
        fetched += future.result()
        done += 1
        if done % 500 == 0 or done == len(files):
            elapsed = time.time() - start
            print(f"{done}/{len(files)} files; fetched {fetched / 1e9:.1f} GB in {elapsed:.0f}s "
                  f"({fetched / 1e6 / max(elapsed, 1):.1f} MB/s)", flush=True)
EOF

for sub in Robots Object Material Eval_Layout; do     # init_assets.sh's completeness check
    [ -d "$ROBODOJO_ASSETS/$sub" ] || { echo "missing $ROBODOJO_ASSETS/$sub" >&2; exit 1; }
done
[ -d "$ROBODOJO_PI05_CKPT/params" ] || { echo "missing $ROBODOJO_PI05_CKPT/params" >&2; exit 1; }
ln -sfn "$ROBODOJO_ASSETS" "$ROBODOJO_ROOT/Assets"    # where RoboDojo's own code looks
du -sh --apparent-size "$ROBODOJO_ASSETS" "$(dirname "$ROBODOJO_PI05_CKPT")"
