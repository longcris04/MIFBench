#!/bin/bash
# Submit one Slurm job per fold (each: 1 GPU, 24 CPUs).
#
#   scripts/submit_folds.sh                 # folds 1-4
#   scripts/submit_folds.sh 2 3             # some folds
#   MIF_SBATCH_ARGS="-p gpu --qos normal" scripts/submit_folds.sh    # your partition / QoS
#   scripts/submit_folds.sh 1 -- --models DATFuse      # run_fold arguments after --
#
# With 4 GPUs available the four folds run at the same time.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/slurm

folds=()
while [[ $# -gt 0 && "$1" != "--" ]]; do folds+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
[[ ${#folds[@]} -eq 0 ]] && folds=(1 2 3 4)

# shellcheck disable=SC2206
extra=(${MIF_SBATCH_ARGS:-})
for fold in "${folds[@]}"; do
  sbatch "${extra[@]}" -J "mif-fold${fold}" scripts/sbatch/fold.sbatch "$fold" "$@"
done
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.20R"
