"""DATFuse adapter for the shared MIF benchmark (mifkit).

Network (``Networks/network.py``, ``MODEL(in_channel=2)`` on the concatenated
pair) and losses (``losses/``) are the published ones; this reproduces the
optimisation of ``Train.py``: Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8) and

    loss = 1*ir + 1*vi + 10*ssim + 100*gra        (Train.py --weight [1, 1, 10, 100])

``Train.py`` itself cannot run (its dataset has ``ir = '...'`` placeholders);
its inputs are ``ToTensor`` images in [0,1] (the Normalize transform it builds
is never applied), which is what the pipeline feeds.

Role mapping: DATFuse's "ir" is the functional modality (CT / PET-Y / SPECT-Y),
its "vi" is MRI.
"""

import torch

from Networks.network import MODEL as net
from losses import gra_loss, ir_loss, ssim_loss, vi_loss

NAME = "DATFuse"
# Batch 32, not the protocol's 64: better Qabf / VIFF / MI / SSIM in every pair than batch 64
# (Train.py: 128).
DEFAULTS = {"batch_size": 32, "lr": 1e-3}
WEIGHTS = (1.0, 1.0, 10.0, 100.0)  # Train.py --weight default
HPARAMS = {
    "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -- Train.py defaults",
    "loss": "1*ir_loss + 1*vi_loss + 10*ssim_loss + 100*gra_loss (Train.py --weight [1,1,10,100])",
    "inputs": "MODEL(in_channel=2) on cat(ir=func Y, vi=MRI), ToTensor scale [0,1]",
    "code_defaults_not_used": "batch 128, 10 epochs (benchmark protocol instead)",
}


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


def build(args, device, pair):
    model = net(in_channel=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"DATFuse: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def train_step(state, batch, args, epoch):
    ir, vi = batch["func"], batch["mri"]
    out = state.model(torch.cat((ir, vi), dim=1))

    loss_ir = WEIGHTS[0] * ir_loss(out, ir)
    loss_vi = WEIGHTS[1] * vi_loss(out, vi)
    loss_ssim = WEIGHTS[2] * ssim_loss(out, ir, vi)
    loss_gra = WEIGHTS[3] * gra_loss(out, ir, vi)
    loss = loss_ir + loss_vi + loss_ssim + loss_gra

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "ir": loss_ir.item(), "vi": loss_vi.item(),
            "ssim": loss_ssim.item(), "gra": loss_gra.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(torch.cat((batch["func"], batch["mri"]), dim=1))


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
