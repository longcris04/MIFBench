"""Run the benchmark of one fold: every model x pair on this job's GPU, scoring on its CPUs.

    python -m mifkit.cli.run_fold --fold 1                      # 20 models x 3 pairs
    python -m mifkit.cli.run_fold --fold 1 --models DATFuse SwinFusion --pairs CT-MRI
    python -m mifkit.cli.run_fold --fold 1 -- --epochs 5        # args after -- go to every train run

Each (pair, model) run is its own ``python -m mifkit.cli.train ... --eval defer``
process, so one model failing (or running out of memory) does not stop the
others, and the GPU is fully released between runs.  The trainer leaves
``results/epoch_NNN/.ready`` after fusing an epoch; a watcher thread here
hands every such epoch to a pool of Octave processes, which keeps scoring
while the GPU trains the next epoch or the next model.

Re-running the same command resumes: runs finished with the current settings
return at once, interrupted ones continue from their last checkpoint, runs
trained with settings that have changed since (batch size, loss, augmentation,
...) are deleted and trained again (``-- --keep-replaced`` keeps them in
``models/<MODEL>/mif_runs_replaced/``), and epochs fused but not yet scored are scored.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time

from ..evaluate import EvalScheduler, scores_path
from ..models import ALL_MODELS, RUN_ORDER
from ..paths import FOLDS, PAIRS, ROOT, RUNS_FOLDER, run_dir
from ..pipeline import READY
from ..reporting import read_epoch_log, rebuild_metrics


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.handle = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def __call__(self, *parts):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] " + " ".join(str(p) for p in parts)
        with self.lock:
            print(line, flush=True)
            self.handle.write(line + "\n")


def pending_epochs(run):
    """Epochs fused (marker present) but not scored."""
    results = os.path.join(run, "results")
    out = []
    if not os.path.isdir(results):
        return out
    for name in sorted(os.listdir(results)):
        marker = os.path.join(results, name, READY)
        if name.startswith("epoch_") and os.path.exists(marker):
            epoch = int(name.split("_")[1])
            if not os.path.exists(scores_path(run, epoch)):
                out.append((epoch, marker))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--models", nargs="*", default=RUN_ORDER, choices=ALL_MODELS,
                        help="models, in the order they are trained (default: the benchmark's, "
                             "VDMUFusion last; models in EXPERIMENTS only when named)")
    parser.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=list(PAIRS))
    parser.add_argument("--order", choices=["pair", "model"], default="model",
                        help="model: every pair of a model, then the next model (default); "
                             "pair: every model on CT-MRI, then PET-MRI, ...")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    parser.add_argument("--eval-workers", type=int, default=max(1, cpus - 6),
                        help="concurrent Octave processes (default: CPUs of the job - 6)")
    parser.add_argument("--eval-chunk", type=int, default=6, help="images per Octave process")
    parser.add_argument("--num-workers", type=int, default=4, help="data-loader workers of each train run")
    parser.add_argument("--retrain", action="store_true", help="pass --fresh: discard existing runs")
    args = parser.parse_args(argv)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(ROOT, "logs", f"fold{args.fold}") if RUNS_FOLDER == "mif_runs" else \
        os.path.join(ROOT, "logs", RUNS_FOLDER, f"fold{args.fold}")
    log = Log(os.path.join(log_dir, "run_fold.log"))
    if args.order == "pair":
        plan = [(pair, model) for pair in args.pairs for model in args.models]
    else:
        plan = [(pair, model) for model in args.models for pair in args.pairs]
    log(f"===== fold {args.fold}: {len(plan)} runs, {args.eval_workers} Octave workers, "
        f"job {os.environ.get('SLURM_JOB_ID')} on {os.uname().nodename}; extra train args: {extra}")

    scheduler = EvalScheduler(workers=args.eval_workers, chunk=args.eval_chunk,
                              on_done=lambda job, rows: rebuild_metrics(job["run"]), log=log)
    submitted = set()
    stop = threading.Event()

    def scan():
        for pair, model in plan:
            run = run_dir(model, pair, args.fold)
            for epoch, marker in pending_epochs(run):
                key = (run, epoch)
                if key in submitted:
                    continue
                try:
                    with open(marker) as handle:
                        info = json.load(handle)
                except (OSError, ValueError):
                    continue
                submitted.add(key)
                scheduler.submit({"pair": pair, "run": run, "epoch": epoch, "stems": info["stems"]})

    def watch():
        while not stop.is_set():
            try:
                scan()
            except Exception as exc:  # noqa: BLE001
                log(f"watcher error: {exc!r}")
            stop.wait(10)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env["PYTHONUNBUFFERED"] = "1"
    statuses = []
    for i, (pair, model) in enumerate(plan, 1):
        run = run_dir(model, pair, args.fold)
        # The trainer decides: a run complete with the current settings returns at once, an
        # interrupted one resumes, one trained with other settings (batch size, augmentation,
        # ...) is deleted (or archived with --keep-replaced) and trained again.
        cmd = [sys.executable, "-m", "mifkit.cli.train", "--model", model, "--pair", pair,
               "--fold", str(args.fold), "--eval", "defer", "--num-workers", str(args.num_workers),
               "--replace-stale"]
        if args.retrain:
            cmd.append("--fresh")
        cmd += extra
        if env.get("TMPDIR"):
            # the node's /dev/shm clean-up after another of our jobs may have removed it (fold.sbatch)
            os.makedirs(env["TMPDIR"], exist_ok=True)
        out_path = os.path.join(log_dir, f"{pair}_{model}.log")
        log(f"[{i}/{len(plan)}] {model} {pair}: start  (log {out_path})")
        started = time.time()
        with open(out_path, "a") as out:
            out.write(f"\n===== {stamp} {' '.join(cmd)}\n")
            out.flush()
            code = subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)
        minutes = (time.time() - started) / 60
        epochs = len(read_epoch_log(run))
        status = "ok" if code == 0 else f"exit {code}"
        log(f"[{i}/{len(plan)}] {model} {pair}: {status} after {minutes:.1f} min, {epochs} epochs trained; "
            f"{scheduler.pending()} epochs waiting for scores")
        statuses.append({"model": model, "pair": pair, "status": status, "minutes": round(minutes, 1),
                         "epochs": epochs})
        scan()

    scan()
    log(f"training finished; waiting for {scheduler.pending()} epochs to be scored")
    while scheduler.pending():
        scheduler.wait(timeout=60)
        scan()
    stop.set()
    scheduler.close()
    for job, error in scheduler.failed():
        log(f"scoring failed: {job['run']} epoch {job['epoch']}: {error}")
    for pair, model in plan:
        rebuild_metrics(run_dir(model, pair, args.fold))
    with open(os.path.join(log_dir, f"status_{stamp}.json"), "w") as handle:
        json.dump({"fold": args.fold, "runs": statuses,
                   "scoring_failed": [(j["run"], j["epoch"], e) for j, e in scheduler.failed()]}, handle, indent=1)
    log("fold finished")


if __name__ == "__main__":
    main()
