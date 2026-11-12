"""Scoring fused images with the MATLAB evaluation code, run under GNU Octave.

``evaluation/octave_eval/mif_eval_list.m`` reproduces the per-image
computation of ``Run_Foder_Updated.m`` (same metrics, same argument order, same
column names).  This module feeds it lists of images and collects the CSVs:

* the images of one epoch are split into chunks, each scored by its own
  ``octave-cli`` process, so ``workers`` CPU cores are busy at once;
* an :class:`EvalScheduler` runs that in background threads, so the GPU can
  train the next epoch while the previous one is scored.

Per image, the inputs are exactly what the MATLAB script reads: path_1 the
functional source as stored (CT / PET / SPECT, colour converted by rgb2gray),
path_2 the MRI source, path_3 the fused grey image.
"""

import csv
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque

from .paths import EVAL_CODE, PAIRS, epoch_dir
from .folds import file_of

#: numeric columns of Run_Foder_Updated.m, in order (Num and Name come first)
METRICS = ["MLI", "SD", "Entropy", "AG", "Qabf", "VIFF", "FMI", "MI", "Xydeas", "Qp", "Q", "Qw",
           "Qe", "NIQE", "SSEQ", "FSIM", "SSIM", "TMQI", "PSNR", "Chen", "CB", "metricWang", "NCIE",
           "metricZhao", "x1", "x2", "Time"]
#: columns that carry no information: metricZhao is disabled (-1), x1/x2 are never
#: filled (0) and Time measures an empty tic/toc in Run_Foder_Updated.m
DEAD_METRICS = {"metricZhao", "x1", "x2", "Time"}
#: lower is better: NIQE and SSEQ (no-reference distortion scores), Chen (Chen-Varshney)
LOWER_IS_BETTER = {"NIQE", "SSEQ", "Chen"}
SCORED_METRICS = [m for m in METRICS if m not in DEAD_METRICS]

OCTAVE_FUNCTION_DIR = os.path.join(EVAL_CODE, "octave_eval")


def octave_binary():
    """octave-cli from $MIF_OCTAVE, else the PATH."""
    path = os.environ.get("MIF_OCTAVE") or shutil.which("octave-cli") or shutil.which("octave")
    if not path:
        raise RuntimeError("GNU Octave not found: set MIF_OCTAVE (scripts/env.sh does)")
    return path


def image_rows(pair, run, epoch, stems):
    """(name, path_1, path_2, path_3) for the scoring list of one epoch."""
    spec = PAIRS[pair]
    out_dir = os.path.abspath(os.path.join(epoch_dir(run, epoch), "gray"))   # Octave runs in a temp dir
    rows = []
    for stem in stems:
        rows.append((stem,
                     os.path.join(spec.image_dir(spec.func), file_of(pair, spec.func, stem)),
                     os.path.join(spec.image_dir(spec.mri), file_of(pair, spec.mri, stem)),
                     os.path.join(out_dir, stem + ".png")))
    return rows


def _octave_env():
    env = dict(os.environ)
    # one core per Octave process: parallelism comes from running several of them
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    return env


def run_octave_chunk(rows, out_csv, log_path=None, timeout=None):
    """Score ``rows`` in one octave-cli process; returns the parsed rows."""
    out_csv = os.path.abspath(out_csv)   # Octave runs in a temporary directory
    work = tempfile.mkdtemp(prefix="mif_eval_")
    try:
        list_file = os.path.join(work, "list.tsv")
        with open(list_file, "w") as handle:
            for row in rows:
                handle.write("\t".join(row) + "\n")
        expr = (f"addpath('{OCTAVE_FUNCTION_DIR}'); "
                f"mif_eval_list('{list_file}', '{out_csv}');")
        cmd = [octave_binary(), "--no-gui", "--norc", "--quiet", "--no-window-system", "--eval", expr]
        with open(log_path or os.devnull, "a") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=_octave_env(),
                                  timeout=timeout, cwd=work)
        if proc.returncode != 0 or not os.path.exists(out_csv):
            raise RuntimeError(f"octave exited with {proc.returncode} (log: {log_path})")
        return read_scores(out_csv)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def read_scores(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_scores(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Num", "Name"] + METRICS)
        writer.writeheader()
        for num, row in enumerate(rows, 1):
            out = {"Num": num, "Name": row["Name"]}
            out.update({m: row.get(m, "") for m in METRICS})
            writer.writerow(out)
    os.replace(tmp, path)


def scores_path(run, epoch):
    return os.path.join(run, "scores", f"epoch_{int(epoch):03d}.csv")


def score_epoch(pair, run, epoch, stems, workers=8, chunk=None, log_dir=None):
    """Score one epoch synchronously with ``workers`` Octave processes; writes scores/epoch_NNN.csv."""
    rows = image_rows(pair, run, epoch, stems)
    missing = [r[3] for r in rows if not os.path.exists(r[3])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} fused images missing, e.g. {missing[0]}")
    size = chunk or max(1, math.ceil(len(rows) / max(1, workers)))
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    os.makedirs(os.path.join(run, "scores"), exist_ok=True)
    log_dir = log_dir or os.path.join(run, "logs")
    os.makedirs(log_dir, exist_ok=True)
    results = [None] * len(chunks)
    errors = []

    def work(i):
        out_csv = os.path.join(run, "scores", f".epoch_{int(epoch):03d}.part{i}.csv")
        try:
            results[i] = run_octave_chunk(chunks[i], out_csv, os.path.join(log_dir, "octave.log"))
        except Exception as exc:  # noqa: BLE001 -- reported below
            errors.append(exc)
        finally:
            if os.path.exists(out_csv):
                os.remove(out_csv)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(chunks))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    merged = [row for part in results for row in part]
    write_scores(scores_path(run, epoch), merged)
    return merged


# --------------------------------------------------------------------- scheduler


class EvalScheduler:
    """Scores epochs in the background with at most ``workers`` Octave processes at once.

    ``submit(job)`` queues the scoring of one epoch; the job is split into
    chunks of ``chunk`` images and every chunk takes one worker slot, so
    several epochs (and runs) can be scored concurrently.  ``on_done(job, rows)``
    is called once an epoch is complete.
    """

    def __init__(self, workers=16, chunk=6, on_done=None, log=print):
        self.workers = max(1, int(workers))
        self.chunk = max(1, int(chunk))
        self.on_done = on_done
        self.log = log
        self._slots = threading.Semaphore(self.workers)
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Condition(self._lock)
        self._queue = deque()
        self._failed = []
        self._dispatcher = threading.Thread(target=self._dispatch, daemon=True)
        self._wake = threading.Event()
        self._closed = False
        self._dispatcher.start()

    # job: dict(pair, run, epoch, stems)
    def submit(self, job):
        with self._lock:
            self._pending += 1
            self._queue.append(job)
        self._wake.set()

    def pending(self):
        with self._lock:
            return self._pending

    def failed(self):
        return list(self._failed)

    def wait(self, timeout=None):
        """Block until every submitted epoch is scored (or failed)."""
        deadline = None if timeout is None else time.time() + timeout
        with self._idle:
            while self._pending:
                remaining = None if deadline is None else max(0.0, deadline - time.time())
                if remaining == 0.0:
                    return False
                self._idle.wait(remaining if remaining is not None else 5.0)
        return True

    def close(self):
        self.wait()
        self._closed = True
        self._wake.set()

    def _dispatch(self):
        while not self._closed:
            self._wake.wait(1.0)
            self._wake.clear()
            while True:
                with self._lock:
                    job = self._queue.popleft() if self._queue else None
                if job is None:
                    break
                threading.Thread(target=self._run_job, args=(job,), daemon=True).start()

    def _run_job(self, job):
        started = time.time()
        try:
            rows = image_rows(job["pair"], job["run"], job["epoch"], job["stems"])
            chunks = [rows[i:i + self.chunk] for i in range(0, len(rows), self.chunk)]
            results = [None] * len(chunks)
            errors = []
            threads = []
            os.makedirs(os.path.join(job["run"], "scores"), exist_ok=True)
            os.makedirs(os.path.join(job["run"], "logs"), exist_ok=True)

            def work(i):
                out_csv = os.path.join(job["run"], "scores",
                                       f".epoch_{int(job['epoch']):03d}.part{i}.csv")
                try:
                    results[i] = run_octave_chunk(chunks[i], out_csv,
                                                  os.path.join(job["run"], "logs", "octave.log"))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                finally:
                    if os.path.exists(out_csv):
                        os.remove(out_csv)
                    self._slots.release()

            for i in range(len(chunks)):
                self._slots.acquire()
                t = threading.Thread(target=work, args=(i,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            merged = [row for part in results for row in part]
            write_scores(scores_path(job["run"], job["epoch"]), merged)
            if self.on_done:
                self.on_done(job, merged)
            self.log(f"scored {os.path.relpath(job['run'])} epoch {job['epoch']} "
                     f"({len(merged)} images, {time.time() - started:.0f}s)")
        except Exception as exc:  # noqa: BLE001 -- a failed epoch must not stop the others
            self._failed.append((job, repr(exc)))
            self.log(f"SCORING FAILED for {job.get('run')} epoch {job.get('epoch')}: {exc!r}")
        finally:
            with self._idle:
                self._pending -= 1
                self._idle.notify_all()
