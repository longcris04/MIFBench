"""Per-run tables: ``metrics_by_epoch.csv`` from the training log and the score files.

The trainer writes ``logs/epochs.json`` (loss terms and timings per epoch) and
the scorer writes ``scores/epoch_NNN.csv`` (one row per test image), possibly
from another process and later.  Whoever finishes something calls
:func:`rebuild_metrics`, which merges both under a file lock into

    metrics_by_epoch.csv   epoch, n_scored, <metric means>, train_<term>, train_seconds, ...
"""

import csv
import fcntl
import json
import math
import os
import re
from contextlib import contextmanager

from .evaluate import LOWER_IS_BETTER, METRICS, read_scores  # noqa: F401  (re-exported)

SCORE_FILE = re.compile(r"^epoch_(\d+)\.csv$")


@contextmanager
def run_lock(run):
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, ".lock"), "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def mean_scores(rows):
    """{metric: mean over images}, NaN / failed values left out."""
    out = {}
    for metric in METRICS:
        values = [v for v in (_float(r.get(metric)) for r in rows) if v is not None]
        out[metric] = sum(values) / len(values) if values else None
    return out


def read_epoch_log(run):
    path = os.path.join(run, "logs", "epochs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return {int(k): v for k, v in json.load(handle).items()}


def write_epoch_log(run, epochs):
    path = os.path.join(run, "logs", "epochs.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({str(k): epochs[k] for k in sorted(epochs)}, handle, indent=1)
    os.replace(tmp, path)


def scored_epochs(run):
    folder = os.path.join(run, "scores")
    found = {}
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = SCORE_FILE.match(name)
            if match:
                found[int(match.group(1))] = os.path.join(folder, name)
    return found


def rebuild_metrics(run):
    """Rewrite metrics_by_epoch.csv; returns its rows."""
    with run_lock(run):
        epochs = read_epoch_log(run)
        scores = scored_epochs(run)
        train_keys, extra_keys = [], []
        for info in epochs.values():
            for key in info.get("train", {}):
                if key not in train_keys:
                    train_keys.append(key)
            for key in info:
                if key != "train" and key not in extra_keys:
                    extra_keys.append(key)
        rows = []
        for epoch in sorted(set(epochs) | set(scores)):
            row = {"epoch": epoch}
            if epoch in scores:
                images = read_scores(scores[epoch])
                row["n_scored"] = len(images)
                row.update(mean_scores(images))
            info = epochs.get(epoch, {})
            for key in train_keys:
                row[f"train_{key}"] = info.get("train", {}).get(key)
            for key in extra_keys:
                row[key] = info.get(key)
            rows.append(row)
        fields = (["epoch", "n_scored"] + METRICS + [f"train_{k}" for k in train_keys]
                  + extra_keys)
        path = os.path.join(run, "metrics_by_epoch.csv")
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("" if v is None else (f"{v:.6g}" if isinstance(v, float) else v))
                                 for k, v in row.items()})
        os.replace(tmp, path)
        return rows
