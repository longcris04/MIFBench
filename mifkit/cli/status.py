"""Progress of the benchmark: epochs trained / scored per run, and the last scores.

    python -m mifkit.cli.status                    # every fold
    python -m mifkit.cli.status --folds 1 --metric Qabf
"""

import argparse
import csv
import os

from ..models import MODELS
from ..paths import FOLDS, PAIRS, run_dir
from ..pipeline import DONE
from ..reporting import read_epoch_log, scored_epochs


def last_metric(run, metric):
    path = os.path.join(run, "metrics_by_epoch.csv")
    if not os.path.exists(path):
        return None, None
    best = (None, None)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(metric):
                best = (int(row["epoch"]), float(row[metric]))
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", nargs="*", type=int, default=list(FOLDS))
    parser.add_argument("--pairs", nargs="*", default=list(PAIRS))
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--metric", default="Qabf")
    args = parser.parse_args(argv)

    for fold in args.folds:
        print(f"== fold {fold}")
        header = f"{'model':<14}" + "".join(f"{pair:>26}" for pair in args.pairs)
        print(header)
        for model in args.models:
            cells = []
            for pair in args.pairs:
                run = run_dir(model, pair, fold)
                trained = len(read_epoch_log(run))
                scored = len(scored_epochs(run))
                epoch, value = last_metric(run, args.metric)
                done = "*" if os.path.exists(os.path.join(run, DONE)) else " "
                text = f"{trained:2d}/{scored:2d}{done}"
                if value is not None:
                    text += f" {args.metric}@{epoch}={value:.3f}"
                cells.append(f"{text:>26}")
            print(f"{model:<14}" + "".join(cells))
    print("cells: epochs trained / epochs scored (* = run finished), then the last scored value")


if __name__ == "__main__":
    main()
