# octave_eval

Runs the metrics of `Run_Foder_Updated.m` under GNU Octave 10.3, on the compute
nodes, for every epoch of every benchmark run. Nothing in `indexes/` or
`octave_compat/` is modified; this folder only adds files.

| file | what it is |
|---|---|
| `mif_eval_list.m` | `Run_Foder_Updated.m`'s per-image loop as a function: takes a list of (name, path_1, path_2, path_3) and writes one CSV row per image, same 29 columns (`Num`, `Name`, `MLI` … `Time`) |
| `compat/dct2.m` | MATLAB's `dct2` (orthonormal 2-D DCT-II). Octave's image package lists it but has no implementation; SSEQ needs it |
| `fast/fmi.m` | `indexes/fmi.m`, memoised (see below) |
| `fast/private/fmi_block_a.m`, `fmi_block_b.m` | the two per-window blocks of `indexes/fmi.m`, copied verbatim |
| `fast/fmi_build_table.m`, `fast/fmi_edge3_table.mat` | the per-window results for every binary 3x3 window pair |
| `fast/Qabf.m` | `indexes/Qabf.m` with the pixel loops written as array expressions |
| `fast/mutual_information.m` | `indexes/mutual_information.m`, histograms counted with `accumarray` |
| `fast/NCC.m` | `imageFusionMetrics-master/NCC.m` (used by `metricWang`, `Q_NCIE`), same |

## Inputs, as in Run_Foder_Updated.m

* `path_1`: the functional source image as stored (CT, or the colour PET/SPECT,
  converted by `rgb2gray`);
* `path_2`: the MRI source (Y of `ConvertRGBtoYUV`);
* `path_3`: the fused **grey** image.

The argument order of every metric is the script's (`im1` = functional,
`im2` = MRI, `imf` = fused), so the asymmetric ones (`FSIM(im1, imf)`,
`psnr(imf, im1)`, `TMQI`) compare the fused image with the functional source,
exactly as there. `metricZhao` stays disabled (-1); `x1`, `x2` (never filled)
and `Time` (an empty `tic/toc`) carry no information.

The one difference from the script: a metric that raises an error gives `NaN`
for that image (with a message on stderr) instead of stopping the run.

## Why `fast/`: speed, with identical numbers

Under Octave (no JIT) the pixel loops of four metric files dominate: about
80 s per 256x256 image, of which `fmi` alone takes 68 s, `Qabf` 5.5 s and
`mutual_information` / `metricWang` / `Q_NCIE` 1.7 s each. The benchmark
scores about 46 000 images per fold (11 models x 20 epochs x 211 test images).

* `fmi(…, 'edge', 3)` evaluates, for every pixel, a block of code that depends
  only on the contents of two 3x3 windows of binary edge maps. `fast/fmi.m`
  runs that same block -- copied verbatim into `private/` -- once per distinct
  window pair and looks the per-pixel values up, then takes `mean2` of the
  same map. For binary 3x3 windows the block has at most 2^18 inputs, so they
  are tabulated once (`fmi_edge3_table.mat`, built in 4 min with
  `fmi_build_table`, 8 processes). Other features / window sizes fall back to
  per-call memoisation.
* `Qabf`, `mutual_information` and `NCC` compute the same element-wise
  expressions over whole arrays (and integer histograms with `accumarray`);
  the original's quirks are kept on purpose -- e.g. `grey_matrix+1` on uint8
  saturates at 255, so grey levels 254 and 255 share a histogram bin.

Checked against the originals on 6 image triplets (CT/PET/SPECT, three kinds of
fused image): `fmi`, `Qabf`, `mutual_information`, `metricWang`, `Q_NCIE`
agree to all 17 printed digits (identical doubles), 0.06-0.14 s instead of
80-100 s for the five together.

`mif_eval_list.m` adds `octave_compat/`, `compat/` and `fast/` to the path only
when it runs under Octave; under MATLAB the original files are used.

## Running it by hand

```bash
source scripts/env.sh
printf '16010\tDataset_clean/CT-MRI/all/CT/16010.bmp\tDataset_clean/CT-MRI/all/MRI/16010.bmp\tfused.png\n' > list.tsv
$MIF_OCTAVE --no-gui --quiet --eval "addpath('matlab_evaluation_code/octave_eval'); mif_eval_list('list.tsv', 'out.csv')"
```

The benchmark calls it through `mifkit/evaluate.py`, a few images per Octave
process and many processes at once.

NIQE comes from `octave_compat/niqe.m` (the published algorithm and pristine
model; not identical to MATLAB's `niqe`, see `octave_compat/README.md`), so
NIQE values are comparable within this benchmark only.
