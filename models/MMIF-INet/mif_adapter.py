"""MMIF-INet adapter for the shared MIF benchmark (mifkit).

The invertible network (``MMIF_INet.py``, ``invblock.py``, ``modules/``), its
initialisation (``model.py``: ``init_model``) and the SSIM / gradient losses
(``pytorch_ssim.py``) are the published ones.  The objective is the one of
``train_logging.py`` with the constants of ``config.py``:

    total = 3 * (1 - ssim(F_Y, MRI_Y, OTH_Y))          l_alpha
          + 1 * grad(F_Y)                               l_beta
          + 1 * (mse(F_Y, MRI_Y) + 0.01 * mse(F_Y, OTH_Y))   l_gamma, mse_w
          + 1 * (mse(F_Cb, OTH_Cb) + mse(F_Cr, OTH_Cr))      l_ks

Optimiser: Adam(betas (0.5, 0.999), eps 1e-6, weight decay 1e-5) as in
``config.py``, constant learning rate (``train_logging.py`` builds a
StepLR(1000, 0.5) on an optimiser it then replaces with a fresh one).

Departure from the released settings: lr 1e-3 instead of ``config.py``'s
10^-4.5 = 3.2e-5, needed for the model to train at all
in the protocol's 20 epochs.

Training uses whole 256x256 images, like every benchmark model (fair
comparison): ``config.py``'s 128x128 crops are not used.  On whole images the
batch size sets the number of optimiser steps, which is what this
initialisation needs.  Benchmark setting: batch 16 at lr 1e-3, under-converged
(Qabf still rising at epoch 20).

``init_model`` draws every weight -- including the scale and shift of the
network's last BatchNorm, which set the output's grey level -- from
N(0, 0.01^2), and Adam moves a weight by about one learning rate per step, so
at 3.2e-5 the output stays near black for ~10^4 steps; the protocol gives
60-100 (the paper trains 1000 epochs).  Fold 1 study, epoch 20, CT-MRI /
PET-MRI near-black pixels in the head and Qabf:

    released settings (60-80 steps)          98 / 95 %   Qabf 0.00 / 0.00
    crops x16 (700-1080 steps), lr 3.2e-5    69 / 46 %        0.04 / 0.04
    lr 1e-3, whole images                    53 / 35 %        0.13 / 0.22
    crops x16, lr 3e-4                       10 / 11 %        0.46 / 0.64
    crops x16, lr 1e-3                        4 /  3 %        0.59 / 0.74   (crops: no longer used)
    whole images, batch 4, lr 1e-3 (4 folds)                 0.59 / 0.73
    whole images, batch 16, lr 3e-3 (4 folds)                0.56 / 0.72
    whole images, batch 16, lr 1e-3 (4 folds, benchmark)     0.42 / 0.57

``config.py`` raises NameError on import (it uses ``torch`` and ``device_ids``
without defining them); it is executed with those two names supplied, the file
itself untouched.

This is the one natively three-channel model: ``datasets_MSRS.py`` loads both
images as RGB (``to_rgb``), cover = MRI (grey replicated), secret = the
functional image as stored (RGB; grey CT replicated), and the network returns
an RGB fusion.  Its luminance is the benchmark's fused image (the metrics
score Y), and the colour result is that Y with the source's Cb/Cr, as for
every other model.
"""

import os
import sys
import types

import torch


def _load_config():
    """Execute the published config.py with the two names it forgets to define."""
    if "config" in sys.modules and getattr(sys.modules["config"], "_mif_patched", False):
        return sys.modules["config"]
    module = types.ModuleType("config")
    module.__dict__["torch"] = torch
    module.__dict__["device_ids"] = [0]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    with open(path) as handle:
        exec(compile(handle.read(), path, "exec"), module.__dict__)
    module._mif_patched = True
    sys.modules["config"] = module
    return module


c = _load_config()

from model import Model, init_model  # noqa: E402  (needs the patched config first)
from pytorch_ssim import gradient, ssim  # noqa: E402

NAME = "MMIF-INet"
# Whole images, no crops; batch 16 (see the docstring):
# at batch 64 the output stays dark after 20 epochs.
DEFAULTS = {"batch_size": 16, "lr": 1e-3}
HPARAMS = {
    "optimizer": f"Adam(lr=1e-3, betas={c.betas}, eps=1e-6, weight_decay={c.weight_decay}), constant LR; "
                 f"config.py's lr is 10^{c.log10_lr}={c.lr:.4g} -- raised so the model trains in 20 epochs",
    "training_images": "whole 256x256 images, one pass per epoch (config.py's 128x128 crops not used)",
    "loss": f"{c.l_alpha}*(1-ssim(F_Y,MRI_Y,OTH_Y)) + {c.l_beta}*grad(F_Y) + "
            f"{c.l_gamma}*(mse(F_Y,MRI_Y) + {c.mse_w}*mse(F_Y,OTH_Y)) + {c.l_ks}*(mse(F_Cb,OTH_Cb)+mse(F_Cr,OTH_Cr))",
    "init": f"init_model: N(0, {c.init_scale}^2) weights, conv5 zero",
    "inputs": "cover = MRI as RGB, secret = functional image RGB; fused = Y of the RGB output",
    "code_defaults_not_used": f"batch {c.batch_size}, {c.epochs} epochs, lr {c.lr:.3g} (benchmark protocol / "
                              "convergence study instead)",
}


def rgb2ycbcr_t(img_rgb):
    """train_logging.py's rgb2ycbcr_t, verbatim (the repo module cannot be imported: it trains on import)."""
    R = img_rgb[:, 0, :, :].unsqueeze(1)
    G = img_rgb[:, 1, :, :].unsqueeze(1)
    B = img_rgb[:, 2, :, :].unsqueeze(1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255
    img_ycbcr = torch.cat([Y, Cb, Cr], axis=1)
    return img_ycbcr, Y, Cb, Cr


def guide_loss(output, bicubic_image):
    return torch.nn.functional.mse_loss(output, bicubic_image)


def Grad_loss(output):
    out_grad = torch.mean(gradient(output), dim=[1, 2, 3])
    out_grad = 1 - torch.mean(out_grad / (out_grad + 1.0))
    return out_grad


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


def build(args, device, pair):
    model = Model().to(device)
    init_model(model)
    model = model.to(device)
    params_trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_trainable, lr=args.lr, betas=c.betas, eps=1e-6,
                                 weight_decay=c.weight_decay)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"MMIF-INet: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def _inputs(batch):
    cover = batch["mri"].repeat(1, 3, 1, 1)      # PIL to_rgb of a grey image
    secret = batch["func_rgb"]
    return cover, secret


def train_step(state, batch, args, epoch):
    cover, secret = _inputs(batch)
    output = state.model(torch.cat((cover, secret), 1))

    FUSION_ycbcr, FUSION_Y, FUSION_Cb, FUSION_Cr = rgb2ycbcr_t(output)
    OTHER_ycbcr, OTHER_Y, OTHER_Cb, OTHER_Cr = rgb2ycbcr_t(secret)
    MRI_Y = cover[:, 0, :, :].unsqueeze(1)
    LOSS_SSIM = 1 - ssim(FUSION_Y, MRI_Y, OTHER_Y)
    mse_chro = guide_loss(FUSION_Cb, OTHER_Cb) + guide_loss(FUSION_Cr, OTHER_Cr)
    mse_Y = guide_loss(FUSION_Y, MRI_Y) + c.mse_w * guide_loss(FUSION_Y, OTHER_Y)
    grad_loss = Grad_loss(FUSION_Y)

    l_loss = c.l_gamma * mse_Y + c.l_ks * mse_chro
    g_loss = grad_loss
    r_loss = LOSS_SSIM
    total_loss = c.l_alpha * r_loss + c.l_beta * g_loss + l_loss
    total_loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad()
    return {"loss": total_loss.item(), "ssim": r_loss.item(), "grad": g_loss.item(),
            "mse_y": mse_Y.item(), "mse_chroma": mse_chro.item()}


@torch.no_grad()
def fuse(state, batch, args):
    cover, secret = _inputs(batch)
    output = state.model(torch.cat((cover, secret), 1))
    _, fused_y, _, _ = rgb2ycbcr_t(output)
    return fused_y


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
