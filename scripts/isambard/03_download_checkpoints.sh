#!/bin/bash
# Download MME-VLA checkpoints and unpack them for evaluation. Login node is fine.
#   03_download_checkpoints.sh                       # pi05 baseline + the Milestone-1 variants
#   03_download_checkpoints.sh perceptual-tokendrop-modul symbolic-grounded-subgoal ...
# Only the requested subfolders are fetched: the full Yinpei/mme_vla_suite repo is ~118 GB.
# Safe to re-run; finished downloads and unpacked checkpoints are skipped.
#
# Unpacked layout (what MME-VLA's loader expects, as produced by upstream scripts/unzip_ckpt.py):
#   $MEMVLA_ROOT/ckpts/<variant>/79999/{params,assets}
#   $MEMVLA_ROOT/ckpts/<variant>/history_config.txt    memory variants only
# The loader reads history_config.txt from the checkpoint's parent folder to build the memory
# module. We store the absolute path of the memory config there (upstream stores a bare file name
# that only resolves from the MME-VLA repo root), so servers can run from any directory.
set -euo pipefail
source "$(dirname "$0")/env.sh"
hf() { uvx --quiet --from huggingface_hub hf "$@"; }

variants=("$@")
[ ${#variants[@]} -gt 0 ] || variants=(perceptual-framesamp-modul recurrent-ttt-expert)

hf download Yinpei/pi05_baseline
for v in "${variants[@]}"; do
    hf download Yinpei/mme_vla_suite --include "$v/*"
done

uv run --quiet --no-project python - "$HF_HOME/hub" "$MEMVLA_ROOT/ckpts" \
    "$MEMVLA_CODE/robomme_policy_learning/src/mme_vla_suite/models/config/robomme" <<'EOF'
import shutil, sys, zipfile
from pathlib import Path

hub, ckpts, memory_configs = map(Path, sys.argv[1:])
for zip_path in sorted(hub.glob("models--Yinpei--*/snapshots/*/*/*.zip")):
    src_dir = zip_path.parent                            # .../<variant>/79999.zip
    dest = ckpts / src_dir.name / zip_path.stem          # ckpts/<variant>/79999
    if not (dest / "params").is_dir():
        partial = dest.with_name(dest.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # Drop the author's absolute path prefix up to and including "79999/"
                parts = Path(member.filename).parts
                rel = Path(*parts[parts.index(zip_path.stem) + 1:]) if zip_path.stem in parts else Path(parts[-1])
                (partial / rel).parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(partial / rel, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)
        partial.rename(dest)
    name_file = src_dir / "history_config.txt"
    if name_file.exists():
        config = memory_configs / name_file.read_text().strip()
        assert config.is_file(), config
        (dest.parent / "history_config.txt").write_text(str(config))   # no trailing newline
    print("ready:", dest)
EOF
