"""ECINFusion adapter for the shared MIF benchmark (mifkit).

``Networks/net.py`` (``Final_MODEL``) and ``losses/`` are the published ones.
This mirrors ``Train.py``: Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8),
forward ``model(ir, vi)`` and

    loss = 1*ssim_ir + 1*ssim_vi + 1*RMI_ir + 2.5*RMI_vi + 1*Hub_ir + 1*Hub_vi
           (Train.py --weight [1, 1, 1, 2.5, 1, 1])

``Train.py`` reads ``./train_old/PET`` (not in the release), so only its data
wiring is replaced.  Its ``GetDataset`` applies ``ToTensor`` only -- the
``RandomCrop(128)`` / ``RandomRotation`` / ``Normalize`` transform it builds is
used just as a not-None flag -- so the authors trained on full images in [0,1],
as the benchmark does.  "ir" is the functional modality (PET in Train.py),
"vi" the MRI (``ir.replace("PET", "MRI")``).

``ChannelTransformer`` is built for 128x128 (one 128x128 patch); at 256x256 the
patch embedding yields a 2x2 grid and the (1,1,16) position embedding
broadcasts over it, which is how the released model also runs on 256x256
test images.
"""

import importlib.machinery
import sys
import types

import torch


def _stub(name, **attrs):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
            module.__dict__.update(attrs)
            sys.modules[name] = module


# net.py imports these visualisation / export tools at module level but never uses them in
# training or inference; stub them only if they are not installed.
_stub("torchsummary", summary=lambda *a, **k: None)
_stub("netron", start=lambda *a, **k: None)
_stub("onnx", shape_inference=types.SimpleNamespace(), load=None, save=None)

from Networks.net import Final_MODEL as net  # noqa: E402
from losses import Hub_ir, Hub_vi, RMI_ir, RMI_vi, ssim_ir, ssim_vi  # noqa: E402

NAME = "ECINFusion"
# Batch 8: Qabf / MI rise as the batch shrinks (48 / 32 / 16 / 8 tried; 64 does not fit an
# 80 GB A100 at 256x256 with the double-precision RMI loss).
DEFAULTS = {"batch_size": 8, "lr": 1e-3}
WEIGHTS = (1.0, 1.0, 1.0, 2.5, 1.0, 1.0)  # Train.py --weight default
HPARAMS = {
    "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -- Train.py defaults",
    "loss": "1*ssim_ir + 1*ssim_vi + 1*RMI_ir + 2.5*RMI_vi + 1*Hub_ir + 1*Hub_vi (Train.py --weight)",
    "inputs": "Final_MODEL()(ir=func Y, vi=MRI), ToTensor scale [0,1], full images",
    "code_defaults_not_used": "batch 6, 800 epochs (benchmark protocol instead); paper: lr 1e-4, 500 epochs",
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
    return f"ECINFusion: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def train_step(state, batch, args, epoch):
    ir, vi = batch["func"], batch["mri"]
    out = state.model(ir, vi)

    l_ssim_ir = WEIGHTS[0] * ssim_ir(out, ir)
    l_ssim_vi = WEIGHTS[1] * ssim_vi(out, vi)
    l_rmi_ir = WEIGHTS[2] * RMI_ir(out, ir)
    l_rmi_vi = WEIGHTS[3] * RMI_vi(out, vi)
    l_hub_ir = WEIGHTS[4] * Hub_ir(out, ir)
    l_hub_vi = WEIGHTS[5] * Hub_vi(out, vi)
    loss = l_ssim_ir + l_ssim_vi + l_rmi_ir + l_rmi_vi + l_hub_vi + l_hub_ir

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "ssim_ir": l_ssim_ir.item(), "ssim_vi": l_ssim_vi.item(),
            "rmi_ir": l_rmi_ir.item(), "rmi_vi": l_rmi_vi.item(),
            "hub_ir": l_hub_ir.item(), "hub_vi": l_hub_vi.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(batch["func"], batch["mri"])


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
