"""Cross-fold tables of the benchmark.

    python -m mifkit.cli.summary                      # every pair and model, last epoch
    python -m mifkit.cli.summary --epoch 10           # tables at epoch 10

For each pair it writes to summary/:

    <PAIR>_by_epoch.csv   model, epoch, n_folds, n_images, then per metric
                          <m>        mean over all test images of the scored folds (out-of-fold:
                                     every image is scored once, by the model of the fold it is a
                                     test image of),
                          <m>_fmean  mean of the per-fold means, <m>_fstd their standard deviation
    <PAIR>_final.csv      the same at the chosen epoch (default: the last epoch every fold has scored)
    <PAIR>_final.md       Markdown: fold mean +- std of the chosen metrics, best per column in bold
"""

import argparse
import csv
import math
import os
import statistics

from ..evaluate import LOWER_IS_BETTER, SCORED_METRICS
from ..models import MODELS
from ..paths import FOLDS, PAIRS, ROOT, run_dir
from ..reporting import read_scores, scored_epochs

TABLE_METRICS = ["Qabf", "VIFF", "FMI", "MI", "SSIM", "FSIM", "PSNR", "NCIE", "CB", "Chen", "SD", "Entropy",
                 "AG", "NIQE", "SSEQ"]


def _float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def collect(model, pair, folds):
    """{epoch: {fold: [per-image rows]}} of every scored epoch."""
    out = {}
    for fold in folds:
        run = run_dir(model, pair, fold)
        for epoch, path in scored_epochs(run).items():
            out.setdefault(epoch, {})[fold] = read_scores(path)
    return out


def aggregate(per_fold, metrics):
    row = {"n_folds": len(per_fold), "n_images": sum(len(r) for r in per_fold.values())}
    for m in metrics:
        pooled = [v for rows in per_fold.values() for v in (_float(r.get(m)) for r in rows) if v is not None]
        fold_means = []
        for rows in per_fold.values():
            vals = [v for v in (_float(r.get(m)) for r in rows) if v is not None]
            if vals:
                fold_means.append(sum(vals) / len(vals))
        row[m] = sum(pooled) / len(pooled) if pooled else None
        row[f"{m}_fmean"] = sum(fold_means) / len(fold_means) if fold_means else None
        row[f"{m}_fstd"] = statistics.stdev(fold_means) if len(fold_means) > 1 else None
    return row


def fmt(v, digits=4):
    return "" if v is None else f"{v:.{digits}g}" if abs(v) >= 1000 else f"{v:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=list(PAIRS))
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--folds", nargs="*", type=int, default=list(FOLDS))
    parser.add_argument("--epoch", type=int, default=None, help="default: last epoch scored in every fold")
    parser.add_argument("--out", default=os.path.join(ROOT, "summary"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    metrics = SCORED_METRICS

    for pair in args.pairs:
        by_epoch, final = [], []
        for model in args.models:
            data = collect(model, pair, args.folds)
            if not data:
                continue
            for epoch in sorted(data):
                by_epoch.append({"model": model, "epoch": epoch, **aggregate(data[epoch], metrics)})
            complete = [e for e in sorted(data) if len(data[e]) == len(args.folds)]
            epoch = args.epoch if args.epoch is not None else (complete[-1] if complete else max(data))
            if epoch in data:
                final.append({"model": model, "epoch": epoch, **aggregate(data[epoch], metrics)})
        if not by_epoch:
            print(f"{pair}: nothing scored yet")
            continue
        fields = ["model", "epoch", "n_folds", "n_images"] + [f"{m}{s}" for m in metrics for s in ("", "_fmean", "_fstd")]
        for name, rows in ((f"{pair}_by_epoch.csv", by_epoch), (f"{pair}_final.csv", final)):
            with open(os.path.join(args.out, name), "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: (fmt(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        # Markdown, fold mean +- std, best (by fold mean) in bold
        shown = [m for m in TABLE_METRICS if m in metrics]
        best = {}
        for m in shown:
            vals = [(r[f"{m}_fmean"], r["model"]) for r in final if r.get(f"{m}_fmean") is not None]
            if vals:
                best[m] = (min if m in LOWER_IS_BETTER else max)(vals)[1]
        lines = [f"# {pair} -- cross-validation ({len(args.folds)} folds), mean +- std over folds", "",
                 "| model | epoch | folds | " + " | ".join(m + (" (lower better)" if m in LOWER_IS_BETTER else "")
                                                         for m in shown) + " |",
                 "|---|---|---|" + "---|" * len(shown)]
        for r in final:
            cells = []
            for m in shown:
                mean, std = r.get(f"{m}_fmean"), r.get(f"{m}_fstd")
                text = "" if mean is None else fmt(mean) + ("" if std is None else f" +- {fmt(std)}")
                cells.append(f"**{text}**" if best.get(m) == r["model"] and text else text)
            lines.append(f"| {r['model']} | {r['epoch']} | {r['n_folds']} | " + " | ".join(cells) + " |")
        with open(os.path.join(args.out, f"{pair}_final.md"), "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print("\n".join(lines) + "\n")
    print(f"tables in {args.out}")


if __name__ == "__main__":
    main()
