"""LRFNet adapter for the shared MIF benchmark (mifkit).

The release has the network (``nets/LRFNet.py``), the data helpers
(``data_process/``) and ``test.py`` -- but no training script and no loss.
The loss combines the two things the release and the paper do state:

* the paper's "detail information-guided SSIM" (DGSSIM, Sec. 3.4, Eq. 11-14),
  ``w1 * (1 - SSIM(I_MRI, I_F)) + w2 * (1 - SSIM(I_Y, I_F))``, with (w1, w2) per
  image pair from the entropy of the Haar high-frequency sub-bands -- the
  repository's own ``haar_weight`` (what ``DatasetFromFolder_w`` returns next
  to every training pair), unchanged;
* the four loss weights of ``test.py``'s training argparse,
  ``--weight [1, 1, 0.0005, 0.00056]``: two order-one weights, read as the two
  SSIM terms, and two of order 5e-4, read as pixel MSE terms -- in the order
  of the sibling repositories' ``Train.py`` (functional term first, MRI second):

    L = 1      * w_y   * (1 - SSIM(I_Y,   I_F)) + 1       * w_mri * (1 - SSIM(I_MRI, I_F))
      + 0.0005 * MSE_255(I_F, I_Y)              + 0.00056 * MSE_255(I_F, I_MRI)

  MSE_255 is the mean squared error on the 0-255 grey scale: on the [0, 1]
  scale these weights would make the two terms ~1e-5 of the loss, i.e. absent.

The paper's loss alone (SSIM only) was tried first: its output, which no
activation bounds, collapsed to near-black / inverted images at every batch
size (fold 1 batch-size study, BENCHMARK.md); the MSE terms anchor the grey
level.

SSIM is the standard Gaussian-window SSIM (11x11, sigma 1.5, C1 = 0.01^2,
C2 = 0.03^2, zero padding), taken per image and weighted per image.

Optimiser and schedule from ``test.py``'s training argparse, which the paper
confirms (Sec. 4.1.4): Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8), batch 4,
100 epochs (the latter two replaced by the benchmark protocol).

Inference as in ``test.py``: ``model(MRI, OTHER_Y)``.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from data_process.Datasets import haar_weight
from nets.LRFNet import mymodel as net

NAME = "LRFNet"
DEFAULTS = {"batch_size": 64, "lr": 1e-3}
#: identifies the loss: a run trained with another recipe is redone (mifkit pipeline)
RECIPE = "dgssim+mse255, test.py --weight [1, 1, 0.0005, 0.00056]"
WEIGHTS = (1.0, 1.0, 0.0005, 0.00056)   # test.py --weight: ssim_y, ssim_mri, mse_y, mse_mri
HPARAMS = {
    "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -- test.py argparse, paper Sec. 4.1.4",
    "loss": "1*w_y*(1-SSIM(Y,F)) + 1*w_mri*(1-SSIM(MRI,F)) + 0.0005*MSE255(F,Y) + 0.00056*MSE255(F,MRI); "
            "(w_mri, w_y) = haar_weight(MRI, Y) (paper's DGSSIM), weights from test.py --weight",
    "inputs": "mymodel()(MRI, func Y) -- test.py",
    "code_defaults_not_used": "batch 4, 100 epochs (benchmark protocol instead)",
    "note": "no training code released: DGSSIM from the paper + the four weights of test.py; "
            "SSIM-only collapsed to black images",
}

WINDOW = 11
SIGMA = 1.5


def sample_extras(func, mri, rgb):
    """haar_weight(MRI, OTHER) on the uint8 images, as DatasetFromFolder_w returns them."""
    w_mri, w_oth = haar_weight(mri, func)
    return {"w_mri": torch.tensor(float(w_mri)), "w_oth": torch.tensor(float(w_oth))}


def _gaussian_window(device, dtype):
    g = torch.tensor([math.exp(-(x - WINDOW // 2) ** 2 / (2 * SIGMA ** 2)) for x in range(WINDOW)],
                     dtype=dtype, device=device)
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).view(1, 1, WINDOW, WINDOW)


def ssim_per_image(x, y):
    """Mean SSIM of each image pair, (B,)."""
    w = _gaussian_window(x.device, x.dtype)
    pad = WINDOW // 2
    mu_x = F.conv2d(x, w, padding=pad)
    mu_y = F.conv2d(y, w, padding=pad)
    sxx = F.conv2d(x * x, w, padding=pad) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return ssim_map.mean(dim=(1, 2, 3))


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


def build(args, device, pair):
    model = net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"LRFNet: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def train_step(state, batch, args, epoch):
    mri, oth = batch["mri"], batch["func"]
    w_mri, w_oth = batch["w_mri"].float(), batch["w_oth"].float()
    out = state.model(mri, oth)
    l_ssim_oth = WEIGHTS[0] * (w_oth * (1 - ssim_per_image(oth, out))).mean()
    l_ssim_mri = WEIGHTS[1] * (w_mri * (1 - ssim_per_image(mri, out))).mean()
    l_mse_oth = WEIGHTS[2] * F.mse_loss(out * 255.0, oth * 255.0)
    l_mse_mri = WEIGHTS[3] * F.mse_loss(out * 255.0, mri * 255.0)
    loss = l_ssim_oth + l_ssim_mri + l_mse_oth + l_mse_mri

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "ssim_oth": l_ssim_oth.item(), "ssim_mri": l_ssim_mri.item(),
            "mse_oth": l_mse_oth.item(), "mse_mri": l_mse_mri.item(), "w_mri": w_mri.mean().item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(batch["mri"], batch["func"])


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
