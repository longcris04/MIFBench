#!/usr/bin/env python3
"""Browse the benchmark's fused images and scores in a web page.

    source scripts/env.sh
    $MIF_PYTHON review_server/server.py                  # http://127.0.0.1:8777
    $MIF_PYTHON review_server/server.py --port 9000
    $MIF_PYTHON review_server/server.py --models ALMFnet LRFNet

Read-only.  Run directories are re-scanned whenever the page asks for a view,
so epochs that a running job writes show up after pressing reload in the page
-- no restart needed.

Views
    The page shows one pair and one fold at a time: ``<PAIR>@<fold>`` with fold
    1-4 (that fold's test split, fused by that fold's models), or
    ``<PAIR>@all``: every image of the pair, each one fused by the model of the
    fold it is a test image of (out-of-fold).  For ``all`` the mean of an epoch
    is pooled over every image, i.e. over the folds that have scored it; the
    training losses are averaged over the folds.

Columns
    One per model (``models/<model>/mif_runs/<pair>/fold<k>``): the benchmark's, then the
    experiments (``mifkit.models.EXPERIMENTS``), each followed by one per variant folder
    of that model, ``models/<model>/mif_runs_<variant>/<pair>/fold<k>``, named
    ``<model>·<variant>`` -- e.g. runs with another learning rate or batch size.
    ``--no-variants`` leaves them out.

Routes
    /                                                 the page (static/)
    /api/index                                        pairs, folds, columns, metrics
    /api/pair/<pair>@<fold>                           every column's epochs, mean and
                                                      per-image scores
    /img/<pair>@<fold>/<column>/<epoch>/<gray|color>/<stem>   a fused image, as PNG
    /src/<pair>@<fold>/<modality>/<stem>              a source image, as PNG
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import threading
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from PIL import Image  # noqa: E402

from mifkit.evaluate import DEAD_METRICS, LOWER_IS_BETTER, METRICS  # noqa: E402
from mifkit.folds import file_of, fold_of, load_folds  # noqa: E402
from mifkit.models import EXPERIMENTS  # noqa: E402
from mifkit.models import MODELS as BENCHMARK_MODELS  # noqa: E402
from mifkit.paths import FOLDS, PAIRS, RUNS_FOLDER, model_dir  # noqa: E402

STATIC_DIR = os.path.join(HERE, "static")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
FUSED_EXT = (".png", ".bmp")

SHOWN_METRICS = [m for m in METRICS if m not in DEAD_METRICS]
#: models whose fusion samples noise (fixed seed per epoch in this pipeline)
STOCHASTIC = {"VDMUFusion"}
FOLD_KEYS = ["all"] + [str(f) for f in FOLDS]

EPOCH_DIR = re.compile(r"^epoch_(\d+)$")
SCORE_FILE = re.compile(r"^epoch_(\d+)(_rerun)?\.csv$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
VIEW = r"([^/@]+)@(all|\d+)"
IMG_ROUTE = re.compile(rf"^/img/{VIEW}/([^/]+)/(\d{{1,4}})/(gray|color)/([^/]+)$")
SRC_ROUTE = re.compile(rf"^/src/{VIEW}/([^/]+)/([^/]+)$")
PAIR_ROUTE = re.compile(rf"^/api/pair/{VIEW}$")
VARIANT_DIR = re.compile(r"^mif_runs_([A-Za-z0-9_-][A-Za-z0-9_.-]*)$")
#: between a model and its variant; the page splits column lists on "," and "column:epoch" on ":"
VARIANT_SEP = "·"

MODELS = list(BENCHMARK_MODELS) + list(EXPERIMENTS)
SHOW_VARIANTS = True
#: column -> (model, run folder under the model's directory); see refresh_columns
COLUMNS = OrderedDict()


class NotFound(Exception):
    pass


# ------------------------------------------------------------------ columns


def _natural(text):
    return [(0, float(t)) if t[0].isdigit() else (1, t) for t in re.findall(r"\d+(?:\.\d+)?|\D+", text)]


def refresh_columns():
    """Re-list every model's mif_runs_<variant> folders; called when the page loads data."""
    global COLUMNS
    found = OrderedDict()
    for model in MODELS:
        found[model] = (model, RUNS_FOLDER)
        if not SHOW_VARIANTS:
            continue
        try:
            names = os.listdir(model_dir(model))
        except FileNotFoundError:
            continue
        variants = [m.group(1) for m in map(VARIANT_DIR.match, names)
                    if m and os.path.isdir(os.path.join(model_dir(model), m.group(0)))]
        for variant in sorted(variants, key=_natural):
            found[f"{model}{VARIANT_SEP}{variant}"] = (model, f"{RUNS_FOLDER}_{variant}")
    COLUMNS = found
    return found


def run_path(column, pair, fold, columns=None):
    model, folder = (columns or COLUMNS)[column]
    return os.path.join(model_dir(model), folder, pair, f"fold{int(fold)}")


def check_view(pair, fold):
    if pair not in PAIRS or fold not in FOLD_KEYS:
        raise NotFound(f"unknown view {pair}@{fold}")


def view_folds(fold):
    return [int(f) for f in FOLD_KEYS[1:]] if fold == "all" else [int(fold)]


# ------------------------------------------------------------------ run data


def _num(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return float(f"{x:.6g}") if math.isfinite(x) else None


def _read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values):
    values = [v for v in values if v is not None]
    return _num(sum(values) / len(values)) if values else None


def view_images(pair, fold):
    """Stems shown for the view: the fold's test split, or every image (all)."""
    data = load_folds(pair)
    if fold == "all":
        return sorted(data["files"][PAIRS[pair].func], key=lambda s: (len(s), s))
    return list(data["folds"][fold]["test"])


def _scan_images(run):
    """{epoch: (fused grey images, mtime_ns of gray/)} for every results/epoch_NNN of a run dir."""
    results = os.path.join(run, "results")
    found = {}
    try:
        entries = os.listdir(results)
    except FileNotFoundError:
        return found
    for entry in entries:
        match = EPOCH_DIR.match(entry)
        if not match:
            continue
        gray = os.path.join(results, entry, "gray")
        try:
            count = sum(f.endswith(FUSED_EXT) for f in os.listdir(gray))
            found[int(match.group(1))] = (count, os.stat(gray).st_mtime_ns)
        except FileNotFoundError:
            continue
    return found


def _score_files(run):
    """{epoch: path}; a *_rerun.csv replaces the training-time file of its epoch."""
    folder = os.path.join(run, "scores")
    chosen = {}
    try:
        entries = sorted(os.listdir(folder))
    except FileNotFoundError:
        return chosen
    for entry in entries:
        match = SCORE_FILE.match(entry)
        if match and (match.group(2) or int(match.group(1)) not in chosen):
            chosen[int(match.group(1))] = os.path.join(folder, entry)
    return chosen


def _mtime(path):
    return os.stat(path).st_mtime_ns if os.path.exists(path) else None


def _signature(pair, fold, columns):
    parts = [tuple(columns)]
    for column in columns:
        for f in view_folds(fold):
            run = run_path(column, pair, f, columns)
            parts.append(tuple(sorted(_scan_images(run).items())))
            files = _score_files(run)
            parts.append(tuple((e, p, os.stat(p).st_mtime_ns) for e, p in sorted(files.items())))
            parts.append(_mtime(os.path.join(run, "metrics_by_epoch.csv")))
            parts.append(_mtime(os.path.join(run, "config.json")))
    return tuple(parts)


def _describe(runs):
    """Settings recorded by the runs, for the column header (the loss of adapters that
    expose it as an option, e.g. ``args.dualbranchcnn.loss``); None for the benchmark's models."""
    for run in runs:
        try:
            with open(os.path.join(run, "config.json")) as handle:
                args = json.load(handle).get("args", {})
        except (OSError, ValueError):
            continue
        experiment = args.get("dualbranchcnn")
        if isinstance(experiment, dict):
            return f"{experiment.get('loss', '?')} loss"
        method = args.get("dgtfusion")
        if not isinstance(method, dict):
            return None
        base = str(method.get("base_loss", "?")).split()[0].rstrip(":")
        weight = method.get("dyn_gt_weight", args.get("dyn_gt_weight"))
        return f"{base} loss · w={weight:g}" if isinstance(weight, (int, float)) else f"{base} loss"
    return None


def _batch_size(runs):
    """Training batch size the runs record in config.json; "a/b" when the folds differ."""
    sizes = set()
    for run in runs:
        try:
            with open(os.path.join(run, "config.json")) as handle:
                size = json.load(handle).get("protocol", {}).get("batch_size")
        except (OSError, ValueError):
            continue
        if size is not None:
            sizes.add(size)
    if not sizes:
        return None
    return sizes.pop() if len(sizes) == 1 else "/".join(str(s) for s in sorted(sizes))


def _load_run(run, stochastic):
    """One fold's run: {epoch: entry} with train stats, image counts, mean and per-image scores."""
    epochs = {}

    def entry(epoch):
        return epochs.setdefault(epoch, {"images": 0, "v": None, "mean": None, "train": {},
                                         "per_image": None, "score_file": None, "stale": False})

    metrics_csv = os.path.join(run, "metrics_by_epoch.csv")
    if os.path.exists(metrics_csv):
        for row in _read_rows(metrics_csv):
            try:
                epoch = int(float(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            item = entry(epoch)
            item["train"] = {k[len("train_"):]: _num(v) for k, v in row.items()
                             if k and k.startswith("train_") and k != "train_seconds"
                             and _num(v) is not None}

    images = _scan_images(run)
    for epoch, (count, mtime_ns) in images.items():
        item = entry(epoch)
        item["images"] = count
        item["v"] = f"{mtime_ns:x}"

    for epoch, path in _score_files(run).items():
        item = entry(epoch)
        item["per_image"] = {row["Name"]: [_num(row.get(m)) for m in SHOWN_METRICS]
                             for row in _read_rows(path) if row.get("Name")}
        item["score_file"] = os.path.basename(path)
        if stochastic and epoch in images:
            item["stale"] = images[epoch][1] > os.stat(path).st_mtime_ns
    for item in epochs.values():
        if item["per_image"]:
            item["mean"] = [_mean(col) for col in zip(*item["per_image"].values())]
    return epochs


def _merge_folds(per_fold):
    """Pool the epochs of several folds: images and scores add up, losses are averaged."""
    merged = {}
    for fold, epochs in per_fold.items():
        for epoch, item in epochs.items():
            m = merged.setdefault(epoch, {"images": 0, "v": None, "mean": None, "train": {}, "_train": [],
                                          "per_image": None, "score_file": None, "stale": False,
                                          "folds_scored": 0, "folds_fused": 0})
            m["images"] += item["images"]
            if item["images"]:
                m["folds_fused"] += 1
            if item["v"]:
                m["v"] = max(m["v"] or "", item["v"])
            if item["per_image"]:
                m["per_image"] = {**(m["per_image"] or {}), **item["per_image"]}
                m["folds_scored"] += 1
            if item["train"]:
                m["_train"].append(item["train"])
            m["stale"] = m["stale"] or item["stale"]
    for epoch, m in merged.items():
        trains = m.pop("_train")
        keys = {k for t in trains for k in t}
        m["train"] = {k: _mean([t.get(k) for t in trains]) for k in keys}
        if m["per_image"]:
            m["mean"] = [_mean(col) for col in zip(*m["per_image"].values())]
            m["score_file"] = f"{m['folds_scored']} of {len(per_fold)} folds"
    return merged


_PAIR_CACHE = {}
_PAIR_LOCK = threading.Lock()


def load_view(pair, fold):
    """The view's payload as JSON bytes, rebuilt only when a file changed."""
    check_view(pair, fold)
    columns = refresh_columns()
    signature = _signature(pair, fold, columns)
    key = f"{pair}@{fold}"
    with _PAIR_LOCK:
        cached = _PAIR_CACHE.get(key)
        if cached and cached[0] == signature:
            return cached[1]
    spec = PAIRS[pair]
    models, settings, batch_sizes = {}, {}, {}
    for column, (model, _) in columns.items():
        runs = {f: run_path(column, pair, f, columns) for f in view_folds(fold)}
        per_fold = {f: _load_run(run, model in STOCHASTIC) for f, run in runs.items()}
        epochs = per_fold[int(fold)] if fold != "all" else _merge_folds(per_fold)
        models[column] = {str(e): epochs[e] for e in sorted(epochs)}
        settings[column] = _describe(runs.values())
        batch_sizes[column] = _batch_size(runs.values())
    payload = {
        "pair": pair,
        "fold": fold,
        "sources": [spec.func, spec.mri],
        "color": spec.color,
        "images": view_images(pair, fold),
        "models": models,
        "settings": settings,
        "batch_sizes": batch_sizes,
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    with _PAIR_LOCK:
        _PAIR_CACHE[key] = (signature, body)
    return body


def index_payload():
    return {
        "models": list(refresh_columns()),
        "pairs": [{"name": name, "func": spec.func, "mri": spec.mri, "color": spec.color}
                  for name, spec in PAIRS.items()],
        "folds": FOLD_KEYS,
        "metrics": SHOWN_METRICS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "stochastic": sorted(STOCHASTIC),
    }


# ------------------------------------------------------------------ images


class PngCache:
    """Images as PNG bytes, kept in memory (LRU by bytes)."""

    def __init__(self, max_bytes=256 << 20):
        self.max_bytes = max_bytes
        self._items = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, path):
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key], key
        if path.lower().endswith(".png"):
            with open(path, "rb") as handle:
                data = handle.read()
        else:
            with Image.open(path) as image:
                buffer = io.BytesIO()
                image.save(buffer, "PNG")
            data = buffer.getvalue()
        with self._lock:
            if key not in self._items:
                self._items[key] = data
                self._bytes += len(data)
                while self._bytes > self.max_bytes and len(self._items) > 1:
                    _, old = self._items.popitem(last=False)
                    self._bytes -= len(old)
        return data, key


PNGS = PngCache()


def fused_path(pair, fold, column, epoch, mode, stem):
    check_view(pair, fold)
    if column not in COLUMNS or not SAFE_NAME.match(stem):
        raise NotFound("no such image")
    if fold == "all":
        fold = fold_of(pair).get(stem)
        if fold is None:
            raise NotFound("no such image")
    run = run_path(column, pair, fold)
    folder = os.path.join(run, "results", f"epoch_{int(epoch):03d}")
    # grey pairs have no colour folder: show the grey image
    for sub in ((mode, "gray") if mode == "color" else ("gray",)):
        for ext in FUSED_EXT:
            path = os.path.join(folder, sub, stem + ext)
            if os.path.isfile(path):
                return path
    raise NotFound("no such image")


def source_path(pair, fold, modality, stem):
    check_view(pair, fold)
    spec = PAIRS[pair]
    if modality not in (spec.func, spec.mri) or not SAFE_NAME.match(stem):
        raise NotFound("no such source image")
    try:
        name = file_of(pair, modality, stem)
    except KeyError:
        raise NotFound("no such source image") from None
    return os.path.join(spec.image_dir(modality), name)


# ------------------------------------------------------------------ HTTP


class Handler(BaseHTTPRequestHandler):
    server_version = "MIFReview/2.0"
    verbose = False

    def do_GET(self):  # noqa: N802 - http.server's naming
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/index":
                return self._json(json.dumps(index_payload()).encode())
            match = PAIR_ROUTE.match(path)
            if match:
                return self._json(load_view(*match.groups()))
            match = IMG_ROUTE.match(path)
            if match:
                return self._png(fused_path(*match.groups()), "v" in query)
            match = SRC_ROUTE.match(path)
            if match:
                return self._png(source_path(*match.groups()), "v" in query)
            raise NotFound(path)
        except NotFound as exc:
            self._send(HTTPStatus.NOT_FOUND, b"not found: " + str(exc).encode(), "text/plain; charset=utf-8")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _static(self, name):
        path = os.path.realpath(os.path.join(STATIC_DIR, name))
        ext = os.path.splitext(path)[1]
        if not path.startswith(STATIC_DIR + os.sep) or ext not in STATIC_TYPES or not os.path.isfile(path):
            raise NotFound(name)
        with open(path, "rb") as handle:
            self._send(HTTPStatus.OK, handle.read(), STATIC_TYPES[ext], {"Cache-Control": "no-cache"})

    def _json(self, body):
        headers = {"Cache-Control": "no-cache"}
        if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024:
            body = gzip.compress(body, compresslevel=5)
            headers["Content-Encoding"] = "gzip"
        self._send(HTTPStatus.OK, body, "application/json", headers)

    def _png(self, path, versioned):
        data, key = PNGS.get(path)
        etag = f'"{key[1]:x}-{key[2]:x}"'
        # A ?v= URL names one version of the file, so the browser may keep it.
        cache = "public, max-age=31536000, immutable" if versioned else "no-cache"
        if self.headers.get("If-None-Match") == etag:
            return self._send(HTTPStatus.NOT_MODIFIED, b"", None, {"ETag": etag, "Cache-Control": cache})
        self._send(HTTPStatus.OK, data, "image/png", {"ETag": etag, "Cache-Control": cache})

    def _send(self, status, body, content_type, headers=None):
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        if self.verbose or (args and str(args[1])[:1] in "45"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default 127.0.0.1: reach it through an SSH/VS Code tunnel)")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"models to show (default: the benchmark's {len(BENCHMARK_MODELS)}, "
                             f"then {', '.join(EXPERIMENTS)})")
    parser.add_argument("--no-variants", action="store_true", help="leave out the <model>/mif_runs_<variant> folders")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    global SHOW_VARIANTS
    if args.models:
        MODELS[:] = args.models
    SHOW_VARIANTS = not args.no_variants
    Handler.verbose = args.verbose

    columns = refresh_columns()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"MIF review server on http://{args.host}:{args.port}  "
          f"({len(MODELS)} models + {len(columns) - len(MODELS)} variant runs, "
          f"{len(PAIRS)} pairs x {len(FOLD_KEYS) - 1} folds)", flush=True)
    for column, (model, folder) in columns.items():
        if folder != RUNS_FOLDER:
            print(f"  {column:<28} {model}/{folder}/<pair>/fold<k>", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
