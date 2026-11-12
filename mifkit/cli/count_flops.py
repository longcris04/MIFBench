"""FLOPs of fusing one 256x256 image pair with a trained model.

    python -m mifkit.cli.count_flops --model DATFuse            # CT-MRI, fold 1

Counted with ``torch.utils.flop_counter.FlopCounterMode`` around one ``adapter.fuse`` call
(batch 1): convolutions, matrix multiplications and attention, 2 FLOPs per multiply-add;
element-wise operations, FFTs and normalisations are not counted. The count does not depend
on the image content, so one pair of one fold is enough. VDMUFusion's count covers its whole
2000-step reverse diffusion.

Writes inference_time/<MODEL>.flops.json.
"""

import argparse
import json
import os
from argparse import Namespace
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from ..data import FusionDataset, pair_data
from ..folds import split_stems
from ..models import ALL_MODELS, load_adapter
from ..paths import PAIRS, run_dir
from ..pipeline import to_device
from .time_inference import OUT_DIR, count_parameters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--pair", default="CT-MRI", choices=list(PAIRS))
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args(argv)

    device = torch.device("cuda")
    adapter = load_adapter(args.model)
    run = run_dir(args.model, args.pair, args.fold)
    blob = torch.load(os.path.join(run, "checkpoints", "last.pt"), map_location=device, weights_only=False)
    run_args = Namespace(**blob["args"])
    state = adapter.build(run_args, device, args.pair)
    adapter.load_state_dict(state, blob["state"])
    adapter.set_mode(state, False)

    data = pair_data(args.pair)
    loader = DataLoader(FusionDataset(data, split_stems(args.pair, args.fold, "test"), train=False), batch_size=1)
    batch = to_device(next(iter(loader)), device)
    counter = FlopCounterMode(display=False)
    torch.manual_seed(run_args.seed + 12345)
    with torch.no_grad(), counter:
        adapter.fuse(state, batch, run_args)
    flops = counter.get_total_flops()
    by_op = sorted(((str(op), int(n)) for op, n in counter.get_flop_counts()["Global"].items()),
                   key=lambda kv: -kv[1])

    out = OrderedDict([("model", args.model), ("pair", args.pair), ("fold", args.fold),
                       ("input", "batch 1, 256x256"), ("flops", int(flops)), ("gflops", flops / 1e9),
                       ("params", count_parameters(state)), ("flops_by_op", OrderedDict(by_op))])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.model}.flops.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"{args.model}: {flops / 1e9:.3f} GFLOPs, {out['params']} parameters -> {path}", flush=True)


if __name__ == "__main__":
    main()
