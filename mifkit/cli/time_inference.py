"""Inference time and parameter count of one trained model.

    python -m mifkit.cli.time_inference --model DATFuse                 # fold 1, all pairs
    python -m mifkit.cli.time_inference --model VDMUFusion --warmup 2

For each pair the model is rebuilt from its run's last checkpoint (epoch 20) and every test
image of the fold is fused one at a time (batch 1, 256x256). The inputs are moved to the GPU
first; the timed span is ``adapter.fuse`` (the network's forward pass plus the model's own
pre/post-processing, as the benchmark fuses) between two ``torch.cuda.synchronize()``.
Image loading, saving and the Cb/Cr merge of colour pairs are not timed. Before timing, the
first test image is fused ``--warmup`` times (cuDNN autotuning, lazy CUDA initialisation).

Parameters = parameters of the trained network the adapter fuses with (``state.model``;
CDDFuse: encoder + decoder + base/detail fusion layers); frozen networks used only by a
training loss (U2Fusion's VGG16) are not counted.

Writes inference_time/<MODEL>.json (per-image times, summary per pair, environment).
"""

import argparse
import json
import os
import platform
import statistics
import time
from argparse import Namespace
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ..color import to_uint8
from ..data import FusionDataset, pair_data
from ..folds import split_stems
from ..models import ALL_MODELS, load_adapter
from ..paths import PAIRS, ROOT, epoch_dir, run_dir
from ..pipeline import to_device

OUT_DIR = os.path.join(ROOT, "inference_time")


def network_modules(state):
    if hasattr(state, "modules_"):          # MMIF-CDDFuse
        return list(state.modules_)
    return [state.model]


def count_parameters(state):
    seen, total = set(), 0
    for module in network_modules(state):
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def time_pair(adapter, model, pair, fold, warmup, device):
    run = run_dir(model, pair, fold)
    blob = torch.load(os.path.join(run, "checkpoints", "last.pt"), map_location=device, weights_only=False)
    args = Namespace(**blob["args"])
    state = adapter.build(args, device, pair)
    adapter.load_state_dict(state, blob["state"])
    adapter.set_mode(state, False)
    epoch = int(blob["epoch"])
    del blob

    data = pair_data(pair)
    stems = split_stems(pair, fold, "test")
    loader = DataLoader(FusionDataset(data, stems, train=False), batch_size=1, shuffle=False, num_workers=0)
    batches = [to_device(batch, device) for batch in loader]
    torch.cuda.synchronize()

    times, diffs = [], []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(args.seed + 12345)
        with torch.no_grad():
            for _ in range(warmup):
                adapter.fuse(state, batches[0], args)
            torch.cuda.synchronize()
            for batch in batches:
                torch.cuda.synchronize()
                started = time.perf_counter()
                fused = adapter.fuse(state, batch, args)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                # sanity check: the image this checkpoint fused during the benchmark
                y = to_uint8(torch.nan_to_num(fused.detach().float(), nan=0.0).clamp(0, 1).cpu().numpy()[0, 0])
                saved = os.path.join(epoch_dir(run, epoch), "gray", batch["name"][0] + ".png")
                if os.path.exists(saved):
                    diffs.append(float(np.abs(y.astype(np.int16) - np.asarray(Image.open(saved), np.int16)).mean()))

    result = OrderedDict([
        ("pair", pair), ("fold", fold), ("epoch", epoch), ("n_images", len(times)), ("warmup", warmup),
        ("mean_s", statistics.fmean(times)), ("std_s", statistics.pstdev(times)),
        ("median_s", statistics.median(times)), ("min_s", min(times)), ("max_s", max(times)),
        ("params", count_parameters(state)),
        ("mean_abs_diff_vs_saved_uint8", statistics.fmean(diffs) if diffs else None),
        ("times_s", times),
    ])
    del state, batches
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30, help="untimed fusions of the first test image")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True          # as in training / fusing (mifkit.pipeline)
    adapter = load_adapter(args.model)

    pairs = []
    for pair in args.pairs:
        result = time_pair(adapter, args.model, pair, args.fold, args.warmup, device)
        print(f"{args.model} {pair}: {result['n_images']} images, mean {result['mean_s'] * 1e3:.3f} ms "
              f"(std {result['std_s'] * 1e3:.3f}, median {result['median_s'] * 1e3:.3f}), "
              f"{result['params']} parameters, mean |diff| vs saved {result['mean_abs_diff_vs_saved_uint8']}",
              flush=True)
        pairs.append(result)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = OrderedDict([
        ("model", args.model),
        ("protocol", "batch 1, 256x256, inputs on the GPU, adapter.fuse between torch.cuda.synchronize() calls; "
                     "I/O and colour merge excluded; cudnn.benchmark on"),
        ("environment", OrderedDict([
            ("gpu", torch.cuda.get_device_name(0)), ("torch", torch.__version__),
            ("cuda", torch.version.cuda), ("cudnn", torch.backends.cudnn.version()),
            ("python", platform.python_version()), ("host", platform.node()),
            ("slurm_job", os.environ.get("SLURM_JOB_ID")),
        ])),
        ("pairs", pairs),
    ])
    path = os.path.join(OUT_DIR, f"{args.model}.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
