#!/bin/bash
# Quick end-to-end check of every adapter: 2 epochs x a few steps, fusing and scoring a few
# test images, into .mif_cache/smoke/ (the real run folders are not touched).
#
#   scripts/smoke_test.sh [models...]                       # on a machine with a GPU
#   srun --gres=gpu:1 -c 24 --mem=64G --time=02:00:00 scripts/smoke_test.sh [models...]   # Slurm
#
# Reports per model: exit status, peak GPU memory and seconds per training step at the
# protocol's batch size, and the time to fuse the images.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cd "$MIF_ROOT"
export TMPDIR="${MIF_TMP_BASE:-/dev/shm}/${USER}_mif_smoke_$$"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

PAIR="${SMOKE_PAIR:-PET-MRI}"
STEPS="${SMOKE_STEPS:-3}"
OUT=.mif_cache/smoke
mkdir -p "$OUT"
models=("$@")
[[ ${#models[@]} -eq 0 ]] && mapfile -t models < <("$MIF_PYTHON" -c "from mifkit.models import RUN_ORDER; print(*RUN_ORDER, sep='\\n')")

for model in "${models[@]}"; do
  echo "=== $model ($PAIR, fold 1, $STEPS steps x 2 epochs) $(date +%T)"
  "$MIF_PYTHON" -m mifkit.cli.train --model "$model" --pair "$PAIR" --fold 1 --epochs 2 \
      --max-steps "$STEPS" --limit-test "${SMOKE_TEST_IMAGES:-8}" --eval inline --eval-workers 8 --eval-chunk 4 \
      --out "$OUT/$model/$PAIR" --fresh ${SMOKE_ARGS:-} > "$OUT/$model.log" 2>&1
  code=$?
  "$MIF_PYTHON" - "$OUT/$model/$PAIR" "$code" <<'EOF'
import json, os, sys, csv
run, code = sys.argv[1], sys.argv[2]
try:
    ep = json.load(open(os.path.join(run, "logs", "epochs.json")))
except Exception:
    print(f"   exit {code}: no epoch finished"); sys.exit()
e = ep[max(ep, key=int)]
steps = max(1, e["steps"])
q = ""
try:
    rows = list(csv.DictReader(open(os.path.join(run, "metrics_by_epoch.csv"))))
    q = " ".join(f"ep{r['epoch']}:Qabf={float(r['Qabf']):.3f}" for r in rows if r.get("Qabf"))
except Exception:
    pass
print(f"   exit {code}: peak {e['peak_mem_gb']:.1f} GB, {e['train_seconds'] / steps:.2f} s/step "
      f"(batch from config), fuse {e['fuse_seconds']:.1f} s, fused range [{e['fused_min']:.3g}, {e['fused_max']:.3g}] {q}")
EOF
  tail -n 3 "$OUT/$model.log" | sed 's/^/   | /'
done
