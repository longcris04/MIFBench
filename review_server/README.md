# Review server

A read-only web page for the benchmark's results: every model's fused images
and scores, at every epoch, side by side -- per fold, or for all folds at once.

```bash
review_server/run.sh                 # http://127.0.0.1:8777
review_server/run.sh --port 9000
review_server/run.sh --no-variants   # only models/<model>/mif_runs, no variant columns
```

It binds to `127.0.0.1`. On a remote machine, forward the port: in VS Code
(Remote-SSH) open the **Ports** panel and forward 8777; from a plain terminal,
`ssh -L 8777:127.0.0.1:8777 <server>`. It needs only the standard library
and PIL (`$MIF_PYTHON` from `scripts/env.sh`).

## Pair and fold

* **Pair**: CT-MRI, PET-MRI, SPECT-MRI.
* **Fold**: `1`-`4` shows that fold's test split (25 % of the pair), fused by
  the models trained on that fold. `All (OOF)` shows every image of the pair,
  each one fused by the model of the fold it is a test image of (out-of-fold):
  187 / 288 / 370 images. Its epoch means are pooled over all images scored
  so far (the `score_file` hint says from how many folds); training losses are
  averaged over the folds.

## What it shows

* **Images**: rows are test images, columns are the two sources and the
  models. Under each fused image is its per-image score on the selected
  metric; ★ marks the best model in the row. Each model column has its own
  epoch selector, and **Best epoch per model** sets each one to the epoch
  with the best mean on the selected metric.
* **Epochs**: one image, models × all epochs, to see how each model's output
  evolves (and where training broke down).
* **Scores**: mean-score table (every metric; best per column shaded blue,
  second best red; rows sorted by the selected metric or by "Σ rank", the sum
  of each model's ranks over the ticked metrics, 1 = best, lowest first),
  the selected metric by epoch for all models, small multiples of every
  metric, and each model's training loss. Tick boxes hide models and metrics;
  the choice is kept in the URL. The metric ticks start at Qabf, FMI, Xydeas,
  Qp, Q, Qw, Qe, SSIM, CB (`DEFAULT_METRICS` in `static/app.js`, button
  "Default"); a browser then keeps its own choice.

Gray / Color: the grey image is what the metrics score (the fused Y); the
colour image of PET-MRI and SPECT-MRI is that Y with the source's Cb/Cr.
CT-MRI has only grey images.

Click any image to open the compare view: all panels zoom and pan together
(scroll / drag), with a per-image score table underneath.

Keys: `[` `]` epoch, `g` / `c` gray / colour, and in the compare view `←` `→`
image, `0` reset zoom, `Esc` close. The view is kept in the URL.

## Variant runs

Every model has a column for `models/<model>/mif_runs/<pair>/fold<k>`. Each folder
`models/<model>/mif_runs_<variant>/` next to it adds a column `<model>·<variant>`
(e.g. DATFuse at another learning rate, written with
`mifkit.cli.train ... --out models/DATFuse/mif_runs_<variant>/<PAIR>/fold<k>`).
Variant folders are listed again whenever the page loads.

## Where the numbers come from

* Per-image scores: `<run>/scores/epoch_NNN.csv` (written by the Octave scorer,
  see `evaluation/octave_eval/`); the epoch mean is their mean.
* Images: `<run>/results/epoch_NNN/{gray,color}/<stem>.png`.
* Training losses: `<run>/metrics_by_epoch.csv`.
* "—" means the epoch has not been scored yet.

**Reload data** re-reads the run directories, so epochs a running job has
just written appear without restarting the server.
