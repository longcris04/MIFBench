"""FATFusion adapter for the shared MIF benchmark (mifkit).

``Networks/networks.py`` (``MODEL``) and ``losses/`` are the published ones.
This mirrors ``Train.py``: Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8),
two-stream forward ``model(ir, vi)`` and

    loss = 0.03*Charbonnier(ir) + 1000*Charbonnier(vi) + 10*tv_ir + 100*tv_vi
           (Train.py --weight [0.03, 1000, 10, 100])

``Train.py`` declares ``--weight-decay 5e-4`` and ``--gamma 0.9`` but passes
neither to the optimiser and builds no scheduler, so there is no weight decay
and no LR schedule.  Its dataset (``ir = ...`` placeholders) feeds ``ToTensor``
images in [0,1]; ``Test.py`` builds ``MODEL(in_channel=1)`` and runs
``model(ir, vi)`` on full images.  "ir" is the functional modality (SPECT in
Test.py), "vi" the MRI.

The paper trains on 120x120 overlapping crops with lr 0.03; the code's lr is
used, and the protocol's full 256x256 images.
"""

import torch

from Networks.networks import MODEL as net
from losses import CharbonnierLoss_IR, CharbonnierLoss_VI, tv_ir, tv_vi

NAME = "FATFusion"
# Two full-resolution Swin streams at 256 channels: at 256x256 batch 12 and above run out of
# memory on an 80 GB A100; 8 peaks at 55.8 GB.
DEFAULTS = {"batch_size": 8, "lr": 1e-3}
WEIGHTS = (0.03, 1000.0, 10.0, 100.0)  # Train.py --weight default
HPARAMS = {
    "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -- Train.py defaults "
                 "(its --weight-decay 5e-4 / --gamma 0.9 are never used)",
    "loss": "0.03*Charbonnier_IR + 1000*Charbonnier_VI + 10*tv_ir + 100*tv_vi (Train.py --weight)",
    "inputs": "MODEL(in_channel=1)(ir=func Y, vi=MRI), ToTensor scale [0,1]",
    "code_defaults_not_used": "batch 8, 10 epochs (benchmark protocol instead); paper: lr 0.03, 120x120 crops",
}


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


def build(args, device, pair):
    model = net(in_channel=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"FATFusion: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def train_step(state, batch, args, epoch):
    ir, vi = batch["func"], batch["mri"]
    out = state.model(ir, vi)

    l_ir = WEIGHTS[0] * CharbonnierLoss_IR(out, ir)
    l_vi = WEIGHTS[1] * CharbonnierLoss_VI(out, vi)
    l_tv_ir = WEIGHTS[2] * tv_ir(out, ir)
    l_tv_vi = WEIGHTS[3] * tv_vi(out, vi)
    loss = l_ir + l_vi + l_tv_ir + l_tv_vi

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "charb_ir": l_ir.item(), "charb_vi": l_vi.item(),
            "tv_ir": l_tv_ir.item(), "tv_vi": l_tv_vi.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(batch["func"], batch["mri"])


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
