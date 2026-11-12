"""The training loop every model goes through.

Protocol (identical for all models; ``config.json`` of each run records it):

* data      -- one fold of ``data/<PAIR>`` (75 % train / 25 % test), images at
               their native 256x256, no crop, no resize;
* augment   -- random horizontal and vertical flips (p = 0.5 each), both modalities alike;
* colour    -- PET / SPECT are fused on their Y channel; the colour result is the fused Y
               with the source's Cb/Cr (``mifkit.color``);
* schedule  -- 20 epochs, batch 64 unless the model does not fit (its adapter's
               ``DEFAULTS["batch_size"]``); learning rate, optimiser and loss are the
               model's own (``HPARAMS`` in its adapter);
* every epoch -- the test split is fused and saved (grey and colour), the checkpoint is
               written, and the images are scored with the MATLAB metrics under Octave,
               in the background (``--eval inline``) or by the fold runner (``--eval defer``).

Adapter interface (``models/<MODEL>/mif_adapter.py``)::

    NAME, DEFAULTS = {"batch_size": int, "lr": float, ...}, HPARAMS = {...}
    build(args, device, pair) -> state
    set_mode(state, train: bool)
    train_step(state, batch, args, epoch) -> {name: float}       (epoch is 1-based)
    fuse(state, batch, args) -> (B,1,H,W) fused luminance in [0,1]
    state_dict(state) / load_state_dict(state, blob)
    optional: add_arguments(parser), prepare(state, args, data, device), sample_extras(func, mri, rgb),
              on_epoch_start(state, epoch, args), on_epoch_end(state, epoch, args),
              describe(state) -> str, current_lr(state) -> float
"""

import argparse
import datetime
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .color import to_uint8, ycbcr_to_rgb
from .data import FusionDataset, pair_data
from .evaluate import EvalScheduler, scores_path
from .folds import split_stems
from .models import ALL_MODELS, load_adapter
from .paths import FOLDS, PAIRS, epoch_dir, model_dir, run_dir
from .reporting import read_epoch_log, rebuild_metrics, write_epoch_log

DEFAULT_BATCH = 64
DEFAULT_EPOCHS = 20
READY = ".ready"
DONE = "DONE"


# --------------------------------------------------------------------------- arguments


def base_parser():
    p = argparse.ArgumentParser(
        description="Train one model on one fold of one pair, fusing and scoring the test split every epoch.")
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--pair", required=True, choices=list(PAIRS))
    p.add_argument("--fold", type=int, required=True, choices=FOLDS)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"default {DEFAULT_BATCH}, or the model's cap when {DEFAULT_BATCH} does not fit")
    p.add_argument("--lr", type=float, default=None, help="default: the learning rate of the model's code")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4, help="data-loader workers")
    flip = p.add_mutually_exclusive_group()
    flip.add_argument("--flip", dest="flip", action="store_true",
                      help="random horizontal + vertical flips (default: on, unless the model's adapter "
                           "switches augmentation off)")
    flip.add_argument("--no-flip", dest="flip", action="store_false", help="no flip augmentation")
    p.set_defaults(flip=None)
    p.add_argument("--repeat", type=int, default=None,
                   help="passes over the training split per epoch (default 1, or the model's own)")
    p.add_argument("--crop", type=int, default=None,
                   help="train on random CROP x CROP patches instead of whole images (default: whole "
                        "images, or the model's own); the test split is always fused whole")
    p.add_argument("--fuse-batch-size", type=int, default=None, help="batch size when fusing the test split")
    p.add_argument("--eval", choices=["inline", "defer", "none"], default="inline",
                   help="inline: score in background Octave processes and wait at the end; "
                        "defer: leave a marker for the fold runner; none: do not score")
    p.add_argument("--eval-workers", type=int, default=16, help="concurrent Octave processes (inline)")
    p.add_argument("--eval-chunk", type=int, default=6, help="images per Octave process")
    p.add_argument("--out", default=None, help="run folder (default models/<MODEL>/mif_runs/<PAIR>/fold<k>)")
    p.add_argument("--fresh", action="store_true", help="discard an existing run in the folder and start over")
    p.add_argument("--replace-stale", action="store_true",
                   help="if the folder holds a run trained with other settings, delete it and start over "
                        "(the fold runner passes this)")
    p.add_argument("--keep-replaced", action="store_true",
                   help="with --replace-stale: move the old run to models/<MODEL>/mif_runs_replaced/<PAIR>/fold<k> "
                        "instead of deleting it")
    p.add_argument("--keep-epoch-checkpoints", action="store_true",
                   help="also keep the weights of every epoch (checkpoints/epoch_NNN.pt)")
    p.add_argument("--max-steps", type=int, default=None, help="debug: optimiser steps per epoch")
    p.add_argument("--limit-test", type=int, default=None, help="debug: fuse only the first N test images")
    return p


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model")
    known, _ = pre.parse_known_args(argv)
    parser = base_parser()
    adapter = None
    if known.model in ALL_MODELS:
        adapter = load_adapter(known.model)
        if hasattr(adapter, "add_arguments"):
            group = parser.add_argument_group(f"{known.model} options")
            adapter.add_arguments(group)
            adapter_args = [a.dest for a in group._group_actions]
        else:
            adapter_args = []
    args = parser.parse_args(argv)
    args.adapter_args = adapter_args if adapter is not None else []
    if adapter is None:
        adapter = load_adapter(args.model)
    return args, adapter


# --------------------------------------------------------------------------- helpers


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.handle = open(path, "a", buffering=1)

    def __call__(self, *parts):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] " + " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.handle.write(line + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _git_head(folder):
    try:
        out = subprocess.run(["git", "-C", folder, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def current_lr(adapter, state):
    if hasattr(adapter, "current_lr"):
        return adapter.current_lr(state)
    optimizer = getattr(state, "optimizer", None)
    if optimizer is not None:
        return optimizer.param_groups[0]["lr"]
    return None


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- run


#: settings that define a run: a checkpoint trained with other values cannot be resumed.
#: "recipe" is the adapter's RECIPE string -- or recipe(args), for an adapter whose loss is an
#: option: bump it when the loss or the training step changes.
COMPARED_ARGS = ("model", "pair", "fold", "batch_size", "lr", "seed", "flip", "repeat", "crop", "recipe")
#: what runs from before a setting existed were trained with
LEGACY_SETTINGS = {"crop": None, "recipe": ""}
_MISSING = object()


def _saved_setting(saved, key):
    if key == "flip" and "flip" not in saved and "no_flip" in saved:   # runs from before --flip existed
        return not saved["no_flip"]
    return saved.get(key, LEGACY_SETTINGS.get(key, _MISSING))


def archive_run(run, args):
    """Move a stale run out of the way: <MODEL>/mif_runs/<PAIR>/fold<k> -> <MODEL>/mif_runs_replaced/<PAIR>/fold<k>.

    The review server shows that folder as a column <MODEL>·replaced (before / after).
    A run in a custom --out folder goes to <out>.replaced-<time> instead.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    standard = os.path.abspath(run_dir(args.model, args.pair, args.fold))
    if os.path.abspath(run) == standard:
        model_root, runs_folder, pair, fold = (os.path.dirname(os.path.dirname(os.path.dirname(standard))),
                                               os.path.basename(os.path.dirname(os.path.dirname(standard))),
                                               os.path.basename(os.path.dirname(standard)),
                                               os.path.basename(standard))
        dest = os.path.join(model_root, f"{runs_folder}_replaced", pair, fold)
        if os.path.exists(dest):
            dest = f"{dest}.{stamp}"
    else:
        dest = f"{run.rstrip(os.sep)}.replaced-{stamp}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(run, dest)
    return dest


def prepare_run_folder(run, args):
    """Start fresh, archive a stale run, or validate a run to resume.

    Returns (checkpoint blob to resume from or None, notes for the log).
    """
    notes = []
    ckpt = os.path.join(run, "checkpoints", "last.pt")
    if args.fresh and os.path.isdir(run):
        for name in ("results", "scores", "checkpoints", "logs"):
            shutil.rmtree(os.path.join(run, name), ignore_errors=True)
        for name in ("metrics_by_epoch.csv", "config.json", DONE):
            if os.path.exists(os.path.join(run, name)):
                os.remove(os.path.join(run, name))
        notes.append("--fresh: the previous run in this folder was removed")
    if not os.path.exists(ckpt):
        return None, notes
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = blob.get("args", {})
    different = {}
    for key in list(COMPARED_ARGS) + list(getattr(args, "adapter_args", [])):
        old = _saved_setting(saved, key)
        if old is _MISSING:
            continue
        new = _jsonable(getattr(args, key, None))
        if old != new:
            different[key] = (old, new)
    if different:
        if not args.replace_stale:
            raise SystemExit(f"{run} holds a run with other settings {different}; pass --replace-stale to "
                             f"archive it and start over, --fresh to delete it, or --out to write elsewhere")
        if getattr(args, "keep_replaced", False):
            dest = archive_run(run, args)
            notes.append(f"settings changed {different}: the previous run was moved to {dest}")
        else:
            shutil.rmtree(run)
            notes.append(f"settings changed {different}: the previous run was deleted")
        return None, notes
    notes.append(f"resuming from {ckpt} (epoch {blob['epoch']} done)")
    return blob, notes


def fuse_split(adapter, state, args, data, loader, out_dir, device, color):
    """Fuse every test image; writes gray/<stem>.png (+ color/<stem>.png for colour pairs)."""
    gray_dir = os.path.join(out_dir, "gray")
    color_dir = os.path.join(out_dir, "color")
    os.makedirs(gray_dir, exist_ok=True)
    if color:
        os.makedirs(color_dir, exist_ok=True)
    count = 0
    stats = {"min": math.inf, "max": -math.inf, "nonfinite": 0}
    # Fixed RNG for fusing: models that sample (VDMUFusion's reverse diffusion, TUFusion's
    # randomly initialised axial attention) see the same noise at every epoch.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        torch.manual_seed(args.seed + 12345)
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                fused = adapter.fuse(state, batch, args)
                fused = fused.detach().float()
                if fused.dim() == 3:
                    fused = fused.unsqueeze(1)
                if fused.shape[1] != 1:
                    raise RuntimeError(f"fuse() returned {tuple(fused.shape)}, expected (B,1,H,W)")
                finite = torch.isfinite(fused)
                stats["nonfinite"] += int((~finite).sum())
                if finite.any():
                    stats["min"] = min(stats["min"], float(fused[finite].min()))
                    stats["max"] = max(stats["max"], float(fused[finite].max()))
                fused = torch.nan_to_num(fused, nan=0.0).clamp_(0, 1).cpu().numpy()[:, 0]
                indices = batch["index"].cpu().numpy() if torch.is_tensor(batch["index"]) else batch["index"]
                for k, name in enumerate(batch["name"]):
                    y = to_uint8(fused[k])
                    Image.fromarray(y).save(os.path.join(gray_dir, name + ".png"))
                    if color:
                        idx = int(indices[k])
                        rgb = ycbcr_to_rgb(y, data.func_cb[idx], data.func_cr[idx])
                        Image.fromarray(rgb).save(os.path.join(color_dir, name + ".png"))
                    count += 1
    return count, stats


def build_config(args, adapter, run, data, train_stems, test_stems, state):
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return OrderedDict([
        ("model", args.model),
        ("pair", args.pair),
        ("fold", args.fold),
        ("run_dir", run),
        ("protocol", OrderedDict([
            ("image_size", list(data.shape)),
            ("crop", args.crop),
            ("augmentation", "random horizontal + vertical flip, p=0.5 each" if args.flip else "none"),
            ("color", "fused on Y of the functional image; colour = fused Y + source Cb/Cr (OpenCV BT.601)"
             if data.spec.color else "grey pair"),
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("lr", args.lr),
            ("repeat", args.repeat),
            ("recipe", args.recipe),
            ("eval_every_epoch", True),
        ])),
        ("hparams", _jsonable(getattr(adapter, "HPARAMS", {}))),
        ("args", _jsonable(vars(args))),
        ("data", OrderedDict([("n_train", len(train_stems)), ("n_test", len(test_stems)),
                              ("train", train_stems), ("test", test_stems)])),
        ("model_description", adapter.describe(state) if hasattr(adapter, "describe") else None),
        ("environment", OrderedDict([
            ("host", platform.node()), ("slurm_job", os.environ.get("SLURM_JOB_ID")),
            ("gpu", gpu), ("python", sys.version.split()[0]), ("torch", torch.__version__),
            ("model_repo_commit", _git_head(model_dir(args.model))),
            ("started", datetime.datetime.now().isoformat(timespec="seconds")),
        ])),
    ])


def train(args, adapter):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device("cuda")
    defaults = getattr(adapter, "DEFAULTS", {})
    if args.batch_size is None:
        args.batch_size = int(defaults.get("batch_size", DEFAULT_BATCH))
    if args.lr is None:
        if "lr" not in defaults:
            raise SystemExit(f"{args.model}: the adapter defines no default learning rate; pass --lr")
        args.lr = float(defaults["lr"])
    if args.fuse_batch_size is None:
        args.fuse_batch_size = int(defaults.get("fuse_batch_size", min(args.batch_size, 32)))
    if args.flip is None:
        args.flip = bool(defaults.get("flip", True))
    if args.repeat is None:
        args.repeat = int(defaults.get("repeat", 1))
    if args.crop is None:
        args.crop = defaults.get("crop")
    args.crop = int(args.crop) if args.crop else None
    recipe = getattr(adapter, "recipe", None)
    args.recipe = str(recipe(args) if callable(recipe) else getattr(adapter, "RECIPE", ""))

    run = os.path.abspath(args.out or run_dir(args.model, args.pair, args.fold))
    os.makedirs(run, exist_ok=True)
    resume, notes = prepare_run_folder(run, args)
    os.makedirs(run, exist_ok=True)
    log = Log(os.path.join(run, "logs", "train.log"))
    log(f"===== {args.model} | {args.pair} | fold {args.fold} | {run}")
    for note in notes:
        log(note)
    if resume is not None and int(resume["epoch"]) >= args.epochs and os.path.exists(os.path.join(run, DONE)):
        missing = [e for e in range(1, args.epochs + 1)
                   if os.path.isdir(os.path.join(epoch_dir(run, e), "gray")) and not os.path.exists(scores_path(run, e))]
        if args.eval == "inline" and missing:
            log(f"complete; scoring {len(missing)} unscored epochs")
            stems = split_stems(args.pair, args.fold, "test")
            scheduler = EvalScheduler(workers=args.eval_workers, chunk=args.eval_chunk,
                                      on_done=lambda job, rows: rebuild_metrics(job["run"]), log=log)
            for epoch in missing:
                present = [s for s in stems if os.path.exists(os.path.join(epoch_dir(run, epoch), "gray", s + ".png"))]
                scheduler.submit({"pair": args.pair, "run": run, "epoch": epoch, "stems": present})
            scheduler.close()
        log(f"complete with these settings ({resume['epoch']} epochs): nothing to train")
        return run

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    data = pair_data(args.pair)
    train_stems = split_stems(args.pair, args.fold, "train")
    test_stems = split_stems(args.pair, args.fold, "test")
    if args.limit_test:
        test_stems = test_stems[:args.limit_test]
    color = data.spec.color

    state = adapter.build(args, device, args.pair)
    if hasattr(adapter, "prepare"):
        adapter.prepare(state, args, data, device)

    start_epoch = 1
    if resume is not None:
        adapter.load_state_dict(state, resume["state"])
        set_rng_state(resume["rng"])
        start_epoch = int(resume["epoch"]) + 1

    config = build_config(args, adapter, run, data, train_stems, test_stems, state)
    with open(os.path.join(run, "config.json"), "w") as handle:
        json.dump(config, handle, indent=1)
    log("settings:", json.dumps(OrderedDict([("protocol", config["protocol"]),
                                             ("hparams", config["hparams"])])))
    if config["model_description"]:
        log(config["model_description"])

    extras = getattr(adapter, "sample_extras", None)
    train_set = FusionDataset(data, train_stems, train=True, flip=args.flip, extras=extras,
                              repeat=args.repeat, crop=args.crop)
    test_set = FusionDataset(data, test_stems, train=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    if resume is not None and "loader_rng" in resume:
        generator.set_state(resume["loader_rng"])
    drop_last = len(train_set) % args.batch_size == 1  # never a lone sample (BatchNorm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=drop_last,
                              num_workers=args.num_workers, pin_memory=True, generator=generator,
                              persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=args.fuse_batch_size, shuffle=False,
                             num_workers=min(2, args.num_workers), pin_memory=True)

    scheduler = None
    if args.eval == "inline":
        scheduler = EvalScheduler(workers=args.eval_workers, chunk=args.eval_chunk,
                                  on_done=lambda job, rows: rebuild_metrics(job["run"]), log=log)
        # epochs fused by an earlier, interrupted process but never scored
        for epoch in range(1, start_epoch):
            if os.path.isdir(os.path.join(epoch_dir(run, epoch), "gray")) and \
                    not os.path.exists(scores_path(run, epoch)):
                scheduler.submit({"pair": args.pair, "run": run, "epoch": epoch, "stems": test_stems})

    steps_log = open(os.path.join(run, "logs", "train_steps.jsonl"), "a", buffering=1)
    epochs_info = read_epoch_log(run)
    global_step = sum(int(v.get("steps", 0)) for k, v in epochs_info.items() if k < start_epoch)
    if start_epoch > args.epochs:
        log(f"all {args.epochs} epochs already trained")

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(adapter, "on_epoch_start"):
            adapter.on_epoch_start(state, epoch, args)
        adapter.set_mode(state, True)
        torch.cuda.reset_peak_memory_stats()
        lr = current_lr(adapter, state)
        sums, counts, steps = OrderedDict(), OrderedDict(), 0
        started = time.time()
        for batch in train_loader:
            batch = to_device(batch, device)
            stats = adapter.train_step(state, batch, args, epoch)
            steps += 1
            global_step += 1
            for key, value in stats.items():
                value = float(value)
                if math.isfinite(value):
                    sums[key] = sums.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1
            if not math.isfinite(float(stats.get("loss", 0.0))):
                log(f"WARNING epoch {epoch} step {steps}: non-finite loss {stats.get('loss')}")
            steps_log.write(json.dumps({"epoch": epoch, "step": steps, "global_step": global_step,
                                        "batch": int(batch["func"].shape[0]),
                                        **{k: float(v) for k, v in stats.items()}}) + "\n")
            if args.max_steps and steps >= args.max_steps:
                break
        torch.cuda.synchronize()
        train_seconds = time.time() - started
        if hasattr(adapter, "on_epoch_end"):
            adapter.on_epoch_end(state, epoch, args)
        peak_train = torch.cuda.max_memory_allocated() / 2 ** 30

        adapter.set_mode(state, False)
        started = time.time()
        out_dir = epoch_dir(run, epoch)
        n_fused, fstats = fuse_split(adapter, state, args, data, test_loader, out_dir, device, color)
        torch.cuda.synchronize()
        fuse_seconds = time.time() - started

        blob = {"epoch": epoch, "state": adapter.state_dict(state), "rng": rng_state(),
                "loader_rng": generator.get_state(), "args": _jsonable(vars(args))}
        os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
        atomic_save(blob, os.path.join(run, "checkpoints", "last.pt"))
        if args.keep_epoch_checkpoints:
            atomic_save({"epoch": epoch, "state": blob["state"]},
                        os.path.join(run, "checkpoints", f"epoch_{epoch:03d}.pt"))

        means = OrderedDict((k, sums[k] / counts[k]) for k in sums)
        epochs_info = read_epoch_log(run)
        epochs_info[epoch] = OrderedDict([
            ("train", means), ("steps", steps), ("lr", lr),
            ("train_seconds", round(train_seconds, 2)), ("fuse_seconds", round(fuse_seconds, 2)),
            ("peak_mem_gb", round(peak_train, 2)), ("fused_min", fstats["min"]),
            ("fused_max", fstats["max"]), ("fused_nonfinite", fstats["nonfinite"]),
            ("finished", datetime.datetime.now().isoformat(timespec="seconds")),
        ])
        write_epoch_log(run, epochs_info)
        rebuild_metrics(run)
        log(f"epoch {epoch}/{args.epochs}: " + " ".join(f"{k}={v:.4g}" for k, v in means.items())
            + f" | {steps} steps, lr {lr if lr is None else f'{lr:.3g}'}, train {train_seconds:.0f}s, "
              f"fuse {n_fused} imgs {fuse_seconds:.0f}s, peak {peak_train:.1f} GB, "
              f"fused range [{fstats['min']:.3g}, {fstats['max']:.3g}]"
            + (f", {fstats['nonfinite']} non-finite px" if fstats["nonfinite"] else ""))

        if args.eval == "inline":
            scheduler.submit({"pair": args.pair, "run": run, "epoch": epoch, "stems": test_stems})
        elif args.eval == "defer":
            with open(os.path.join(out_dir, READY), "w") as handle:
                json.dump({"pair": args.pair, "epoch": epoch, "stems": test_stems}, handle)

    steps_log.close()
    if scheduler is not None:
        log(f"waiting for {scheduler.pending()} epochs to be scored")
        scheduler.close()
        for job, error in scheduler.failed():
            log(f"scoring failed: epoch {job['epoch']}: {error}")
    rebuild_metrics(run)
    with open(os.path.join(run, DONE), "w") as handle:
        handle.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
    log("done")
    return run
