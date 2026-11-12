"""(Re)score fused images that are already on disk.

    python -m mifkit.cli.score --model DATFuse --pair CT-MRI --fold 1             # unscored epochs
    python -m mifkit.cli.score --model DATFuse --pair CT-MRI --fold 1 --all       # every epoch again
    python -m mifkit.cli.score --models all --pairs all --folds 1 2 --workers 20  # many runs

Writes scores/epoch_NNN.csv and rebuilds metrics_by_epoch.csv of each run.
"""

import argparse
import os
import re

from ..evaluate import EvalScheduler, scores_path
from ..folds import split_stems
from ..models import MODELS
from ..paths import FOLDS, PAIRS, epoch_dir, run_dir
from ..reporting import rebuild_metrics

EPOCH_DIR = re.compile(r"^epoch_(\d+)$")


def fused_epochs(run):
    results = os.path.join(run, "results")
    if not os.path.isdir(results):
        return []
    return sorted(int(m.group(1)) for m in map(EPOCH_DIR.match, os.listdir(results)) if m)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "--models", dest="models", nargs="+", required=True,
                        help="model names or 'all'")
    parser.add_argument("--pair", "--pairs", dest="pairs", nargs="+", required=True, help="pairs or 'all'")
    parser.add_argument("--fold", "--folds", dest="folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--epochs", nargs="*", type=int, default=None, help="default: every fused epoch")
    parser.add_argument("--all", action="store_true", help="also rescore epochs that have scores")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--chunk", type=int, default=6)
    parser.add_argument("--out", default=None, help="a single run folder instead of model/pair/fold")
    args = parser.parse_args(argv)
    models = MODELS if args.models == ["all"] else args.models
    pairs = list(PAIRS) if args.pairs == ["all"] else args.pairs

    scheduler = EvalScheduler(workers=args.workers, chunk=args.chunk,
                              on_done=lambda job, rows: rebuild_metrics(job["run"]))
    n = 0
    for model in models:
        for pair in pairs:
            for fold in args.folds:
                run = args.out or run_dir(model, pair, fold)
                stems = split_stems(pair, fold, "test")
                for epoch in (args.epochs or fused_epochs(run)):
                    if not os.path.isdir(os.path.join(epoch_dir(run, epoch), "gray")):
                        continue
                    if os.path.exists(scores_path(run, epoch)) and not args.all:
                        continue
                    present = [s for s in stems
                               if os.path.exists(os.path.join(epoch_dir(run, epoch), "gray", s + ".png"))]
                    scheduler.submit({"pair": pair, "run": run, "epoch": epoch, "stems": present})
                    n += 1
    print(f"scoring {n} epochs with {args.workers} Octave workers")
    scheduler.close()
    for job, error in scheduler.failed():
        print(f"FAILED {job['run']} epoch {job['epoch']}: {error}")


if __name__ == "__main__":
    main()
