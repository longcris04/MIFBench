#!/bin/bash
# FLOPs of one fusion (256x256, batch 1) for trained models, one process per model.
#
#   sbatch -J mif-flops scripts/sbatch/count_flops.sbatch DATFuse SwinFusion ...
#
# Results: inference_time/<MODEL>.flops.json
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
cd "$MIF_ROOT"
export TMPDIR="${MIF_TMP_BASE:-/dev/shm}/${USER}_mif_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "job $SLURM_JOB_ID on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(date)"
for model in "$@"; do
    "$MIF_PYTHON" -m mifkit.cli.count_flops --model "$model" || echo "FAILED: $model"
done
echo "done $(date)"
