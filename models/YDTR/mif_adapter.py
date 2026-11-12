"""YDTR adapter for the shared MIF benchmark (mifkit).

Network (``Networks/net.py``, ``MODEL(in_channel=1)`` as ``Test.py`` builds it) and
losses (``losses/__init__.py``) are the published ones; this reproduces the
optimisation of ``Train.py``: Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8) and

    loss = 1*ssim_ir + 0.05*ssim_vi + 0.0006*sf_ir + 0.00025*sf_vi
                                            (Train.py --weight [1, 0.05, 0.0006, 0.00025])

``Train.py`` cannot run as released (its dataset has ``ir = '...'`` placeholders); its
inputs are ``ToTensor`` images in [0,1] (the Normalize transform it builds is never
applied), which is what the pipeline feeds.  Inference as ``Test.py``:
``model(ir, vi)``; the tanh output is clipped to [0,1] by the pipeline.

Role mapping, as for DATFuse (same authors, same training template): YDTR's "ir" is
the functional modality (CT / PET-Y / SPECT-Y), its "vi" is MRI.

``losses.sf`` hard-codes ``cuda:0``; a job sees only its own GPU, which is cuda:0.
"""

import torch

from Networks.net import MODEL as net
from losses import sf_loss_ir, sf_loss_vi, ssim_loss_ir, ssim_loss_vi

NAME = "YDTR"
# at 256x256 batch 48 and above run out of memory on an 80 GB A100; 40 peaks at 67.3 GB (Train.py: 32)
DEFAULTS = {"batch_size": 40, "lr": 1e-3}
WEIGHTS = (1.0, 0.05, 0.0006, 0.00025)  # Train.py --weight default
HPARAMS = {
    "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -- Train.py defaults",
    "loss": "1*ssim_loss_ir + 0.05*ssim_loss_vi + 0.0006*sf_loss_ir + 0.00025*sf_loss_vi "
            "(Train.py --weight [1, 0.05, 0.0006, 0.00025])",
    "inputs": "MODEL(in_channel=1)(ir=func Y, vi=MRI), ToTensor scale [0,1] -- Test.py",
    "code_defaults_not_used": "batch 32, 10 epochs, 120x120 training images (benchmark protocol instead)",
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
    return f"YDTR: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def train_step(state, batch, args, epoch):
    ir, vi = batch["func"], batch["mri"]
    out = state.model(ir, vi)

    loss_ssim_ir = WEIGHTS[0] * ssim_loss_ir(out, ir)
    loss_ssim_vi = WEIGHTS[1] * ssim_loss_vi(out, vi)
    loss_sf_ir = WEIGHTS[2] * sf_loss_ir(out, ir)
    loss_sf_vi = WEIGHTS[3] * sf_loss_vi(out, vi)
    loss = loss_ssim_ir + loss_ssim_vi + loss_sf_ir + loss_sf_vi

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "ssim_ir": loss_ssim_ir.item(), "ssim_vi": loss_ssim_vi.item(),
            "sf_ir": loss_sf_ir.item(), "sf_vi": loss_sf_vi.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(batch["func"], batch["mri"])


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
