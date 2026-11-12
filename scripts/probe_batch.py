"""Largest training batch that fits on this GPU, for one model, at the protocol's 256x256.

    srun ... $MIF_PYTHON scripts/probe_batch.py FATFusion [MixFuse ...]

For each candidate batch size (64 first, then smaller) it builds the model fresh,
runs two optimiser steps on real PET-MRI images and fuses one batch; the first
size that runs without CUDA out-of-memory is reported with its peak memory.

Several models are probed one per process (the repositories' top-level module names
-- ``model``, ``net``, ``utils`` ... -- collide).  ``MIF_PROBE_EPOCHS`` (default "3")
lists the epochs whose training steps are run: for a model trained in stages
(``on_epoch_start``), e.g. ``MIF_PROBE_EPOCHS=1,11`` steps both stages of a
10 + 10-epoch schedule and reports the larger peak.
"""

import gc
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mifkit.data import FusionDataset, pair_data  # noqa: E402
from mifkit.folds import split_stems  # noqa: E402
from mifkit.models import load_adapter  # noqa: E402
from mifkit.pipeline import base_parser, to_device  # noqa: E402

CANDIDATES = [64, 56, 48, 40, 32, 24, 16, 12, 8, 6, 4, 2]
EPOCHS = [int(e) for e in os.environ.get("MIF_PROBE_EPOCHS", "3").split(",")]


def probe(model, pair="PET-MRI"):
    adapter = load_adapter(model)
    data = pair_data(pair)
    stems = split_stems(pair, 1, "train")
    device = torch.device("cuda")
    for bs in CANDIDATES:
        args = base_parser().parse_args(["--model", model, "--pair", pair, "--fold", "1"])
        if hasattr(adapter, "add_arguments"):
            import argparse
            extra = argparse.ArgumentParser()
            adapter.add_arguments(extra)
            for k, v in vars(extra.parse_args([])).items():
                setattr(args, k, v)
        args.batch_size, args.lr = bs, adapter.DEFAULTS["lr"]
        state = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            state = adapter.build(args, device, pair)
            if hasattr(adapter, "prepare"):
                adapter.prepare(state, args, data, device)
            ds = FusionDataset(data, stems[:bs], train=True, extras=getattr(adapter, "sample_extras", None))
            batch = torch.utils.data.default_collate([ds[i] for i in range(bs)])
            batch = to_device(batch, device)
            t = time.time()
            for epoch in EPOCHS:
                if hasattr(adapter, "on_epoch_start"):
                    adapter.on_epoch_start(state, epoch, args)
                adapter.set_mode(state, True)
                for _ in range(2):
                    adapter.train_step(state, batch, args, epoch)
            torch.cuda.synchronize()
            step = (time.time() - t) / (2 * len(EPOCHS))
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            adapter.set_mode(state, False)
            with torch.no_grad():
                if model != "VDMUFusion":
                    adapter.fuse(state, batch, args)
            fuse_peak = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"{model}: batch {bs} fits -- train peak {peak:.1f} GB, {step:.2f} s/step, "
                  f"with fuse peak {fuse_peak:.1f} GB", flush=True)
            return bs
        except torch.OutOfMemoryError:
            print(f"{model}: batch {bs} -> out of memory", flush=True)
        finally:
            del state
            gc.collect()
            torch.cuda.empty_cache()
    return None


if __name__ == "__main__":
    if len(sys.argv) > 2:
        for name in sys.argv[1:]:
            subprocess.call([sys.executable, os.path.abspath(__file__), name])
    elif len(sys.argv) == 2:
        probe(sys.argv[1])
