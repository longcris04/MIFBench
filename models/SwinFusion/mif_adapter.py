"""SwinFusion adapter for the shared MIF benchmark (mifkit).

Network, loss and optimiser are the release's medical-fusion setting
(``options/swinir/train_swinfusion_med.json``, ``models/model_plain.py``):

* ``netG``: ``define_G`` with net_type ``swinfusion1``, in_chans 1, img_size 128,
  window 8, embed_dim 60, num_heads 6, mlp_ratio 2, no upsampler, ``1conv``, default init;
* loss ``fusion_loss_med`` (``models/loss_med.py``): ``20 * L1(F, max(A, B))
  + 100 * L1(|grad F|, max(|grad A|, |grad B|)) + 50 * (1 - (SSIM(A, F) + SSIM(B, F)) / 2)``;
* Adam(lr 2e-4, weight decay 0); the MultiStepLR milestones (250k+ iterations) are never
  reached, so the rate is constant;
* inputs as the ``med`` dataset: A = MRI, B = the functional image's Y.

Fusion as ``test_swinfusion.py``: the EMA copy ``netE`` (``E_decay``) of the generator,
on the image padded to the next multiple of the window by appending its mirror image
(256 -> 264 rows and columns), output cropped back and clipped to [0, 1].

EMA decay: the release keeps ``netE = 0.999 * netE + 0.001 * netG`` after every step and
publishes the models of iteration 10000, where the initial weights' share of netE is
0.999^10000 = 4.5e-5.  A benchmark run has ~360-700 steps (20 epochs at batch 8), over
which a 0.999 average would still be 50-70 % the random initialisation; the decay is set
per run so that the initial weights keep the same share at the last step:
``decay = 0.999 ** (10000 / total_steps)`` (0.973 for 360 steps, 0.986 for 700).

Benchmark deviations: whole 256x256 images (release: 64x64 crops), batch 8 -- the largest
that fits (release 12-16 over 3 GPUs), 20 epochs instead of 10000 iterations.
"""

import copy
import math

import torch

from mifkit.folds import split_stems
from models.loss_med import fusion_loss_med
from models.select_network import define_G

NAME = "SwinFusion"
# Swin blocks on all 65536 pixels (patch size 1): at 256x256 batch 12 and above run out of memory on an
# 80 GB A100; 8 peaks at 63.7 GB (release: 12-16)
DEFAULTS = {"batch_size": 8, "lr": 2e-4}
E_DECAY, CODE_ITERS = 0.999, 10000          # train_swinfusion_med.json, released 10000_E.pth
WINDOW = 8
NETG = {"net_type": "swinfusion1", "upscale": 1, "in_chans": 1, "img_size": 128, "window_size": WINDOW,
        "img_range": 1.0, "depths": [6, 6, 6, 6], "embed_dim": 60, "num_heads": [6, 6, 6, 6], "mlp_ratio": 2,
        "upsampler": None, "resi_connection": "1conv", "init_type": "default", "init_bn_type": "uniform",
        "init_gain": 0.2}
HPARAMS = {
    "optimizer": "Adam(lr=2e-4, weight_decay=0), constant (milestones 250k+ never reached) -- "
                 "train_swinfusion_med.json",
    "loss": "fusion_loss_med: 20*L1(F, max) + 100*L1(grad F, max grad) + 50*(1 - (SSIM(A,F)+SSIM(B,F))/2)",
    "network": NETG,
    "inputs": "A = MRI, B = func Y (med dataset)",
    "fusion": "EMA model netE, input mirror-padded to 264x264 as test_swinfusion.py, output cropped, clipped",
    "ema": "decay 0.999**(10000/total_steps): the initial weights keep the share they have at the release's "
           "iteration 10000 (0.999**10000) -- E_decay 0.999 alone would leave netE 50-70 % random after "
           "the run's ~360-700 steps",
    "code_defaults_not_used": "64x64 crops, batch 12-16, 10000 iterations (benchmark protocol instead)",
}


class State:
    def __init__(self, net_g, net_e, optimizer, criterion):
        self.net_g = net_g
        self.net_e = net_e
        self.optimizer = optimizer
        self.criterion = criterion
        self.decay = E_DECAY


def build(args, device, pair):
    net_g = define_G({"netG": dict(NETG), "is_train": True}).to(device)
    net_e = copy.deepcopy(net_g).eval()          # update_E(0): netE starts as netG
    for p in net_e.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(net_g.parameters(), lr=args.lr, weight_decay=0)
    return State(net_g, net_e, optimizer, fusion_loss_med().to(device))


def prepare(state, args, data, device):
    n = len(split_stems(args.pair, args.fold, "train")) * max(1, args.repeat or 1)
    steps = n // args.batch_size if n % args.batch_size == 1 else math.ceil(n / args.batch_size)
    total = max(1, steps * args.epochs)
    state.decay = E_DECAY ** (CODE_ITERS / total)


def describe(state):
    n = sum(p.numel() for p in state.net_g.parameters())
    return f"SwinFusion: {n / 1e6:.3f}M parameters; EMA decay {state.decay:.4f}"


def set_mode(state, train):
    state.net_g.train(train)
    state.net_e.eval()


@torch.no_grad()
def _update_e(state):
    """model_base.update_E: netE = decay * netE + (1 - decay) * netG (parameters)."""
    params_g = dict(state.net_g.named_parameters())
    for k, p in state.net_e.named_parameters():
        p.data.mul_(state.decay).add_(params_g[k].data, alpha=1 - state.decay)


def train_step(state, batch, args, epoch):
    a, b = batch["mri"], batch["func"]
    state.optimizer.zero_grad()
    fused = state.net_g(a, b)
    total, loss_grad, loss_l1, loss_ssim = state.criterion(a, b, fused)
    total.backward()
    state.optimizer.step()
    _update_e(state)
    return {"loss": total.item(), "grad": loss_grad.item(), "l1": loss_l1.item(), "ssim": loss_ssim.item()}


def _mirror_pad(x):
    """test_swinfusion.py: pad to (h // window + 1) * window by appending the flipped image."""
    h, w = x.shape[-2:]
    hp = (h // WINDOW + 1) * WINDOW
    wp = (w // WINDOW + 1) * WINDOW
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :hp, :]
    return torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :wp]


@torch.no_grad()
def fuse(state, batch, args):
    a, b = batch["mri"], batch["func"]
    h, w = a.shape[-2:]
    out = state.net_e(_mirror_pad(a), _mirror_pad(b))
    return out[..., :h, :w]


def state_dict(state):
    return {"net_g": state.net_g.state_dict(), "net_e": state.net_e.state_dict(),
            "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.net_g.load_state_dict(blob["net_g"])
    state.net_e.load_state_dict(blob["net_e"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
