#!/bin/bash
# Create the two environments of the benchmark (run once; needs internet).
#
#   mifbench   Python 3.11 + PyTorch 2.6.0 (CUDA 12.4) and every model's dependencies
#   mifoctave  GNU Octave 10.3 (conda-forge) + the Octave Forge "image" package (the metrics)
#
#   scripts/setup_env.sh                 # both
#   MIF_SKIP_OCTAVE=1 scripts/setup_env.sh
#
# Then: source scripts/env.sh (it finds both environments under `conda info --base`/envs).
# If pip is set up with an extra index that cannot be reached, PIP_CONFIG_FILE=/dev/null
# makes it ignore that configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA="$(command -v mamba || command -v conda)"

# ---------------------------------------------------------------- PyTorch environment
"$CONDA" create -y -n mifbench -c conda-forge --override-channels python=3.11 pip
PY="$(conda run -n mifbench python -c 'import sys; print(sys.executable)')"
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -r requirements.txt
"$PY" -m pip check

[[ -n "${MIF_SKIP_OCTAVE:-}" ]] && exit 0

# ---------------------------------------------------------------- Octave environment
"$CONDA" create -y -n mifoctave -c conda-forge --override-channels octave=10.3.0 cxx-compiler c-compiler \
    fortran-compiler make pkg-config
OCT="$(conda run -n mifoctave printenv CONDA_PREFIX)"
# mkoctfile remembers the compilers of the conda-forge build machine: point it at the env's own.
export OCTAVE_HOME="$OCT" CC="$OCT/bin/x86_64-conda-linux-gnu-cc" CXX="$OCT/bin/x86_64-conda-linux-gnu-c++" \
    F77="$OCT/bin/x86_64-conda-linux-gnu-gfortran" DL_LD="$OCT/bin/x86_64-conda-linux-gnu-c++" \
    LD_CXX="$OCT/bin/x86_64-conda-linux-gnu-c++"
"$OCT/bin/octave-cli" --no-gui --eval "pkg install -forge image; pkg list"

# SSEQ predicts with libsvm's svmpredict MEX: build it for this Octave
# (evaluation/indexes/SSEQ-main/libsvm-master/matlab/svmpredict.mex ships prebuilt for Octave 10.3, x86-64)
(cd evaluation/indexes/SSEQ-main/libsvm-master/matlab && \
    "$OCT/bin/mkoctfile" --mex -I.. svmpredict.c ../svm.cpp svm_model_matlab.c -o svmpredict.mex && \
    rm -f ./*.o ../*.o) || echo "WARNING: svmpredict.mex was not rebuilt; the shipped one is used" >&2
