# octave_compat

Drop-in replacements that let the evaluation code run under GNU Octave on the
HPC server, where MATLAB is not installed. Nothing in `indexes/` was modified.

Add this folder to the path **only when running under Octave**:

```matlab
addpath('octave_compat');   % Octave only -- see the warning below
```

> **Do not add this folder to the path in MATLAB.** `niqe.m` would shadow
> MATLAB's own `niqe()` and silently change your numbers.

## What is here and why

| File | Replaces | Reason |
|---|---|---|
| `niqe.m` + `niqe_modelparameters.mat` | IPT `niqe()` | Octave has no NIQE at all |
| `blkproc.m` | removed MATLAB `blkproc()` | used by `metricChen`, `TMQI`; Octave ships `blockproc` instead |
| `freqspace.m` | MATLAB `freqspace()` | used by `metricChenBlum`; not in Octave |
| `normcdf.m`, `normpdf.m`, `betapdf.m` | Statistics Toolbox | used by `TMQI`; the Octave `statistics` package needs Octave >= 11.1 |

`indexes/SSEQ-main/libsvm-master/matlab/svmpredict.mex` was compiled from the
bundled libsvm sources so `SSEQ` works on Linux. The original Windows
`.mexw64` files are untouched next to it.

## NIQE: read this before comparing numbers

`niqe.m` implements the published algorithm of Mittal, Soundararajan & Bovik,
"Making a Completely Blind Image Quality Analyzer", IEEE SPL 20(3), 2013,
with the authors' own pristine model.

**It does not reproduce MATLAB's `niqe()` exactly.** MathWorks fitted their own
pristine multivariate-Gaussian model on a different set of images, and that
model ships only inside MATLAB. The algorithm is the same; the reference
statistics are not, so the absolute values differ.

Consequences:

- Re-score **every** image with this implementation. Never put NIQE numbers
  from MATLAB and from here in the same table or the same comparison.
- Rankings between fusion methods are what remain meaningful.

Verified against an independent Python implementation (BasicSR) driven by the
same pristine parameters: agreement within 0.005 absolute on 256x256 test
images (< 0.02% relative). The residual gap comes from `fspecial('gaussian')`
differing in the tenth decimal between Octave and MATLAB, amplified because a
256x256 image yields only 4 patches, making the fitted covariance rank
deficient. That sensitivity is inherent to NIQE on small images.

The pristine model parameters (`mu_prisparam` 1x36, `cov_prisparam` 36x36) were
cross-checked bit-for-bit against two independent redistributions (scikit-video
and BasicSR) before being saved here.

### Behaviour worth knowing

- Images smaller than 96x96 raise an error -- NIQE has no meaningful score there.
- A perfectly flat image returns `NaN` with a warning: no patch passes the
  sharpness filter, so there is nothing to fit.
- Integer images are read on their native 0-255 scale; `double` images are
  assumed to be in [0,1], matching MATLAB's convention.
- The published sharpness threshold of 0.75 is the default. Pass a fourth
  argument to override it (`-Inf` keeps every patch, which is what BasicSR does).
