#!/bin/bash
# One fold of the benchmark: 20 models x 3 pairs on one GPU, Octave scoring on the CPUs.
#
#   sbatch -J mif-fold1 scripts/sbatch/fold.sbatch 1            # or: scripts/submit_folds.sh
#   sbatch -J mif-fold1 scripts/sbatch/fold.sbatch 1 --models DATFuse SwinFusion
#   sbatch -p <partition> --qos <qos> ...                        # your cluster's partition / QoS
#
# Arguments after the fold number go to `python -m mifkit.cli.run_fold`.
# Resubmitting the same fold resumes it (finished runs are skipped).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
FOLD="${1:?usage: sbatch scripts/sbatch/fold.sbatch <fold 1-4> [run_fold args]}"
shift

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
cd "$MIF_ROOT"
# temporary files (data-loader workers, Octave lists) in memory; MIF_TMP_BASE overrides /dev/shm
export TMPDIR="${MIF_TMP_BASE:-/dev/shm}/${USER}_mif_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "job ${SLURM_JOB_ID:-local} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$MIF_PYTHON" -m mifkit.cli.run_fold --fold "$FOLD" "$@"
