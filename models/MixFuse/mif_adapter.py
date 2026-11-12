"""MixFuse adapter for the shared MIF benchmark (mifkit).

The release has the network (``Network/MixFusion_Cross.py``), the loss
(``loss/loss_fusion.py``) and the dataset (``data_utils/msrs_data.py``), but
no training or test script, and the paper is not among ``papers/``.

* Loss: the released ``fusion_loss_vif`` (10*gradient + 5*max-intensity L1 + 4*SSIM).
* Roles: ``msrs_data.py`` reads the ``MRI`` folder into the infrared slot and
  the colour modality's Y into the visible slot; ``forward(vis_image, ir_image)``.
* Network built for the input size (``MixFusion(img_size=256)``; window 8).
* Optimiser -- not in the code: Adam(lr 2e-4, betas (0.9, 0.999)), the setting
  of SwinFusion (``papers/SwinFusion.pdf``, Sec. IV-B), the SwinIR-based design
  MixFuse builds on and compares with (same window size 8); constant LR.
"""

import torch

from Network.MixFusion_Cross import MixFusion
from loss.loss_fusion import fusion_loss_vif

NAME = "MixFuse"
# Three cross/self-attention stages at full resolution: at 256x256 batch 16 and above run out of
# memory on an 80 GB A100; 12 peaks at 72.9 GB.
DEFAULTS = {"batch_size": 12, "lr": 2e-4}
HPARAMS = {
    "optimizer": "Adam(lr=2e-4, betas=(0.9, 0.999)), constant -- NOT in the release (no train script, "
                 "paper not in papers/): SwinFusion's setting (papers/SwinFusion.pdf Sec. IV-B)",
    "loss": "fusion_loss_vif (loss/loss_fusion.py): 10*grad + 5*l1(max) + 4*ssim",
    "inputs": "MixFusion(img_size=256)(vis=func Y, ir=MRI) -- msrs_data.py roles",
    "note": "msrs_data.py resizes training images to 224x224; the protocol's native 256x256 is used",
}


class State:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion


def build(args, device, pair):
    model = MixFusion(img_size=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = fusion_loss_vif().to(device)
    return State(model, optimizer, criterion)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"MixFuse: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def train_step(state, batch, args, epoch):
    vis_y, ir = batch["func"], batch["mri"]
    fused = state.model(vis_y, ir)
    loss, loss_grad, loss_l1, loss_ssim = state.criterion(vis_y, ir, fused)

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "grad": loss_grad.item(), "l1": loss_l1.item(), "ssim": loss_ssim.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(batch["func"], batch["mri"])


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
