# Source me:   source scripts/env.sh
# Paths of the benchmark's two environments (created by scripts/setup_env.sh).
# Override any of them before sourcing, e.g. MIF_PYTHON=/path/to/python.

export MIF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${MIF_CONDA_ENVS:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    MIF_CONDA_ENVS="$(conda info --base)/envs"
  else
    MIF_CONDA_ENVS="$HOME/.conda/envs"
  fi
fi
[[ -d "$MIF_CONDA_ENVS/mifbench" ]] || { [[ -d "$HOME/.conda/envs/mifbench" ]] && MIF_CONDA_ENVS="$HOME/.conda/envs"; }

# PyTorch environment: every model, the pipeline, the review server
export MIF_PYTHON="${MIF_PYTHON:-$MIF_CONDA_ENVS/mifbench/bin/python}"

# GNU Octave 10.3 (conda-forge) + Octave Forge "image" package: the MATLAB metrics
export MIF_OCTAVE_PREFIX="${MIF_OCTAVE_PREFIX:-$MIF_CONDA_ENVS/mifoctave}"
export OCTAVE_HOME="$MIF_OCTAVE_PREFIX"
export MIF_OCTAVE="${MIF_OCTAVE:-$MIF_OCTAVE_PREFIX/bin/octave-cli}"

export PYTHONPATH="$MIF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
