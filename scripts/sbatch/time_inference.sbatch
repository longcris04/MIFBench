#!/bin/bash
# Inference time + parameter count of trained models (fold-1 checkpoints), one model after another
# on one GPU, one process per model (the repositories' module names collide).
#
#   sbatch -J mif-time scripts/sbatch/time_inference.sbatch DATFuse SwinFusion ...
#   sbatch -J mif-time-vdm scripts/sbatch/time_inference.sbatch VDMUFusion -- --warmup 2
#
# Arguments after "--" go to every `python -m mifkit.cli.time_inference` call.
# Results: inference_time/<MODEL>.json
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
MODELS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do MODELS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
cd "$MIF_ROOT"
export TMPDIR="${MIF_TMP_BASE:-/dev/shm}/${USER}_mif_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "job $SLURM_JOB_ID on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for model in "${MODELS[@]}"; do
    "$MIF_PYTHON" -m mifkit.cli.time_inference --model "$model" "$@" || echo "FAILED: $model"
done
echo "done $(date)"
