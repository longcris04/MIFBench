"""ALMFnet adapter for the shared MIF benchmark (mifkit).

The searched architecture (``model_DAG.py`` + genotype ``DAG_lat`` of
``genotypes.py``, 16 initial channels) and the published ``FusionLoss``
(``loss.py``: 1.2*pixel + 0.5*grad + 0.3*seg) are used unchanged; only the
step-3 loop of ``train.py`` is re-expressed, because its ``TrainingData``
reads five folders per sample (``mri``, ``oth``, ``mri_mask``, ``oth_mask``,
``seg``) of which the benchmark data has only the first two.

The three supervision maps, rebuilt from the sources at every step
-------------------------------------------------------------------
``mri_mask`` / ``oth_mask``
    the paper's pixel-loss weight (Sec. III-C, Eq. 6-7): ``M = 0.5 + (v_m - v_i)/2``
    with ``v`` the histogram visual-saliency map ``v(x) = sum_y |I(x) - I(y)|``,
    scaled to [0,1] by its maximum; ``oth_mask = 1 - M``.  On the eight samples
    the repository ships in ``data/train`` this reproduces ``mri_mask``
    with a correlation of 0.88-0.95 (0 at background = 0.5, as there).
``seg``
    the paper takes it from an MRI tissue segmentation [57]; the shipped
    ``data/train/seg`` maps are binary masks of the bright MRI structures.
    Here: the brightest class of a 3-class Otsu threshold of the MRI
    (IoU 0.57-0.76 with the shipped maps; a plain 2-class Otsu gives 0.27-0.68).

Hyper-parameters (``train.py``): Adam, lr = ``--learning_rate`` default 0.025,
no weight decay.  The paper reports lr 1e-4 and batch 64; the code's value is
used, as for every model.  Batch / epochs / crop follow the benchmark protocol
(64 / 20 / full 256x256) instead of the code's 8 / 300 / 128 random crops.
"""

import torch

import genotypes
from loss import FusionLoss
from model_DAG import Network_DAG as Network

NAME = "ALMFnet"
DEFAULTS = {"batch_size": 64, "lr": 0.025}
HPARAMS = {
    "optimizer": "Adam(lr=0.025) -- train.py --learning_rate default; torch default betas/eps, no weight decay",
    "loss": "FusionLoss (loss.py): 1.2*pixel + 0.5*grad + 0.3*seg",
    "architecture": "Network_DAG(16, genotypes.DAG_lat)",
    "masks": "mri_mask = 0.5 + (v_mri - v_oth)/2 (paper Eq. 6-7, v = histogram saliency / max), "
             "oth_mask = 1 - mri_mask; seg = top class of 3-class Otsu on MRI (rebuilt every step)",
    "inputs": "model(mri, oth): MRI and the Y channel of CT/PET/SPECT (eval.py: YCrCb Y of both)",
    "code_defaults_not_used": "batch 8, 300 epochs, 128x128 random crops (benchmark protocol instead); "
                              "paper: lr 1e-4, batch 64",
}

ARCH = "DAG_lat"
INIT_CHANNELS = 16
LEVELS = 256


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()


def build(args, device, pair):
    genotype = eval(f"genotypes.{ARCH}")
    model = Network(INIT_CHANNELS, genotype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"ALMFnet {ARCH}: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def _histograms(img):
    """(B,1,H,W) in [0,1] -> grey levels (B,HW) and their histograms (B,256)."""
    b = img.shape[0]
    idx = torch.clamp((img.reshape(b, -1) * (LEVELS - 1)).round().long(), 0, LEVELS - 1)
    hist = torch.zeros(b, LEVELS, device=img.device, dtype=torch.float64)
    hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
    return idx, hist


def saliency(img):
    """Visual saliency v(x) = sum_y |I(x) - I(y)| through the histogram, divided by its maximum."""
    idx, hist = _histograms(img)
    levels = torch.arange(LEVELS, device=img.device, dtype=torch.float64)
    dist = (levels[:, None] - levels[None, :]).abs()
    s_level = hist @ dist.t()                                   # (B, 256): saliency of each grey level
    sal = torch.gather(s_level, 1, idx)
    sal = sal / sal.amax(dim=1, keepdim=True).clamp_min(1e-12)
    return sal.reshape_as(img).to(img.dtype)


def otsu3_top(img):
    """Mask of the brightest class of a 3-class Otsu threshold (exhaustive over grey-level pairs)."""
    idx, hist = _histograms(img)
    p = hist / hist.sum(dim=1, keepdim=True)
    levels = torch.arange(LEVELS, device=img.device, dtype=torch.float64)
    w = torch.cumsum(p, dim=1)                       # P(level <= t)
    mu = torch.cumsum(p * levels, dim=1)             # sum_{level <= t} level * p
    w_t, mu_t = w[:, -1:], mu[:, -1:]
    w0, m0 = w[:, :, None], mu[:, :, None]                            # class 0: [0, t1]
    w1, m1 = w[:, None, :] - w[:, :, None], mu[:, None, :] - mu[:, :, None]  # (t1, t2]
    w2, m2 = w_t[:, :, None] - w[:, None, :], mu_t[:, :, None] - mu[:, None, :]  # (t2, 255]
    score = m0 ** 2 / w0 + m1 ** 2 / w1 + m2 ** 2 / w2                  # between-class variance + const
    t1 = torch.arange(LEVELS, device=img.device)
    valid = (t1[None, :] > t1[:, None])[None] & (w0 > 0) & (w1 > 0) & (w2 > 0)
    score = torch.where(valid, score, torch.full_like(score, -float("inf")))
    best = score.reshape(score.shape[0], -1).argmax(dim=1)
    t2 = (best % LEVELS).view(-1, 1)
    return (idx > t2).to(img.dtype).reshape_as(img)


def supervision_maps(mri, oth):
    mri_mask = (0.5 + (saliency(mri) - saliency(oth)) / 2).clamp(0, 1)
    return mri_mask, 1.0 - mri_mask, otsu3_top(mri)


def train_step(state, batch, args, epoch):
    mri, oth = batch["mri"], batch["func"]
    with torch.no_grad():
        mri_mask, oth_mask, seg = supervision_maps(mri, oth)
    fus = state.model(mri, oth)
    state.optimizer.zero_grad()
    loss = FusionLoss(fus, mri, oth, mri_mask, oth_mask, seg, state.l1, state.mse)
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(batch["mri"], batch["func"])


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
