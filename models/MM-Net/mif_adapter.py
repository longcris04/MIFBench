"""MM-Net adapter for the shared MIF benchmark (mifkit).

The release (``yuliu316316/MM-Net-Fusion``) has the network (``net/``), the loss terms
(``loss.py``: ``intensity_loss``, ``ssim``, ``FocalFrequencyLoss``), ``config.py`` and
``test.py`` -- no training script.  Training is written from the paper
(``papers_MIF/MM-Net.pdf``, Sec. III-F and IV-A) with the released loss terms:

    L_total = L_spa + lambda * L_fre,                                   lambda = 300
    L_spa   = L_ssim + alpha * L_int,                                   alpha  = 10
    L_ssim  = w_A (1 - SSIM(F, I_A)) + w_B (1 - SSIM(F, I_B))           loss.ssim (per image)
    L_int   = w_A ||F - I_A||_1 + w_B ||F - I_B||_1                     loss.intensity_loss
    L_fre   = FFL(F, I_A) + beta * FFL(F, I_B),                         beta   = 0.1
    m_X     = mean( (sign(I_X - tau) + 1) / 2 * I_X ),  (w_A, w_B) = softmax(m_A / c, m_B / c)
                                                                        tau = 0.85, c = 0.5

Adam, learning rate fixed at 1e-4 (paper; ``config.py`` initial_learning_rate), weight
decay 1e-4 (``config.py`` "L2_norm"); the paper's batch 32 / 50 epochs on 128x128 patches
are replaced by the benchmark protocol.

Inputs and output as ``test.py``: I_A = MRI, I_B = the functional image (Y); the network
predicts a weight map ``W`` (sigmoid) and the fused image is ``W * I_A + (1 - W) * I_B``.
"""

import torch

from loss import FocalFrequencyLoss, intensity_loss, ssim
from net.fusionnet import FusionNet

NAME = "MM-Net"
# MixFormer at 256x256: batch 64 runs out of memory on an 80 GB A100; 56 peaks at 75.8 GB (paper: 32)
DEFAULTS = {"batch_size": 56, "lr": 1e-4}
WEIGHT_DECAY = 1e-4            # config.py: config["L2_norm"] = 1e-4
ALPHA, BETA, LAMBDA = 10.0, 0.1, 300.0
TAU, C = 0.85, 0.5
HPARAMS = {
    "optimizer": "Adam(lr=1e-4 fixed, weight_decay=1e-4) -- paper Sec. IV-A, config.py "
                 "(initial_learning_rate, L2_norm); no training code released",
    "loss": "L_ssim + 10*L_int + 300*(FFL(F,MRI) + 0.1*FFL(F,func)), saliency weights tau 0.85, c 0.5 -- "
            "paper Eq. (7)-(14), terms from loss.py",
    "inputs": "FusionNet(MRI, func Y) -> W; F = W*MRI + (1-W)*func -- test.py",
    "code_defaults_not_used": "batch 32, 50 epochs, 128x128 patches with stride 20 (benchmark protocol instead)",
}


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.ffl = FocalFrequencyLoss()


def build(args, device, pair):
    model = FusionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"MM-Net: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def _fuse(model, img_a, img_b):
    w = model(img_a, img_b)
    return w * img_a + (1 - w) * img_b


def saliency_weights(img_a, img_b):
    """Eq. (8)-(9): mean intensity of the salient pixels (> tau), softmax over the pair at scale c."""
    m_a = ((torch.sign(img_a - TAU) + 1) / 2 * img_a).mean(dim=(1, 2, 3))
    m_b = ((torch.sign(img_b - TAU) + 1) / 2 * img_b).mean(dim=(1, 2, 3))
    w = torch.softmax(torch.stack([m_a, m_b], dim=1) / C, dim=1)
    return w[:, 0], w[:, 1]


def train_step(state, batch, args, epoch):
    img_a, img_b = batch["mri"], batch["func"]
    fused = _fuse(state.model, img_a, img_b)
    w_a, w_b = saliency_weights(img_a, img_b)
    l_ssim = (w_a * ssim(fused, img_a) + w_b * ssim(fused, img_b)).mean()    # loss.ssim returns 1 - SSIM
    l_int = (w_a * intensity_loss(fused, img_a) + w_b * intensity_loss(fused, img_b)).mean()
    l_spa = l_ssim + ALPHA * l_int
    l_fre = state.ffl(fused, img_a) + BETA * state.ffl(fused, img_b)
    loss = l_spa + LAMBDA * l_fre

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "ssim": l_ssim.item(), "int": l_int.item(), "fre": l_fre.item(),
            "w_mri": w_a.mean().item()}


@torch.no_grad()
def fuse(state, batch, args):
    return _fuse(state.model, batch["mri"], batch["func"])


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
