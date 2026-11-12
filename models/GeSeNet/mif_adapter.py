"""GeSeNet adapter for the shared MIF benchmark (mifkit).

The release has the network (``model.py``, ``RMM.py``) and ``test.py`` -- no training
script and no loss.  Training is written from the paper (``papers_MIF/GeSeNet.pdf``,
Sec. III-E and IV-A):

    L_total = L_E + alpha * L_SSIM (+ beta * L_S),     alpha = 0.5, beta = 0.3
    L_E     = L_c + gamma * L_g,                        gamma = 10
    L_c     = ||F - max(I_a, I_b)||_1 / (H W)
    L_g     = || |grad F| - max(|grad I_a|, |grad I_b|) ||_1 / (H W)   (Sobel, |Gx| + |Gy|)
    L_SSIM  = (1 - SSIM(F, I_a)) + (1 - SSIM(F, I_b))

Adam(lr 1e-3, weight decay 2e-4) (paper; batch 4 and 500 epochs on 24x24 patches are
replaced by the benchmark protocol).

Not reproduced -- the semantic loss L_S: it comes from a "pre-trained semantic module"
(the bilateral attention decoder of the paper's ref. [39], trained with SGD after the
fusion network) whose code and weights are not released, and the paper does not say
what it was pre-trained on or where the one-hot targets ``I_o`` of Eq. (16)-(17) come
from for unlabelled Harvard images.  Per the benchmark's rule for what the paper does not
describe (follow APLFusion, which has no semantic module), the run has beta = 0.

Inputs as ``test.py``: ``model(image_vis, image_ir)`` with vis = the functional image
(CT / PET / SPECT; the network reads its Y channel only) and ir = MRI.

Inference as ``test.py``: the model is left in training mode (``test.py`` never calls
``model.eval()``; the region mask module's eval path, the sparse convolution, also
returns a tensor that ``GeSeNet.forward`` indexes as a tuple), so the masks are drawn
with Gumbel noise -- under the pipeline's fixed fusion seed, the same noise every epoch.
The fused image is min-max normalised per image, as ``test.py`` does.
"""

import kornia
import torch
import torch.nn.functional as F

from model import GeSeNet

NAME = "GeSeNet"
# batch: the protocol's 64, peak 51.1 GB (paper: 4)
DEFAULTS = {"batch_size": 64, "lr": 1e-3}
WEIGHT_DECAY = 2e-4
ALPHA, BETA, GAMMA = 0.5, 0.3, 10.0
HPARAMS = {
    "optimizer": "Adam(lr=1e-3, weight_decay=2e-4) -- paper Sec. IV-A (no training code released)",
    "loss": "L_E + 0.5*L_SSIM; L_E = L1(F, max(a,b)) + 10*L1(|grad F|, max(|grad a|,|grad b|)) (Sobel); "
            "L_SSIM = (1-SSIM(F,a)) + (1-SSIM(F,b)) -- paper Eq. (9)-(13)",
    "not_reproduced": "semantic loss 0.3*L_S (pre-trained semantic module and its targets not released or "
                      "described) -> beta = 0, as APLFusion (benchmark rule for undescribed parts)",
    "inputs": "GeSeNet(vis = func Y, ir = MRI) -- test.py",
    "fusion": "training-mode forward (as test.py, which never calls eval()), per-image min-max",
    "code_defaults_not_used": "batch 4, 500 epochs, 24x24 patches with stride 8 (benchmark protocol instead)",
}


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


def build(args, device, pair):
    model = GeSeNet(output=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"GeSeNet: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def _sobel(x):
    g = kornia.filters.spatial_gradient(x, mode="sobel", normalized=False)   # (B,1,2,H,W)
    return g[:, :, 0].abs() + g[:, :, 1].abs()


def _ssim(a, b):
    return kornia.metrics.ssim(a, b, window_size=11, max_val=1.0).mean()


def loss_fn(fused, img_a, img_b):
    l_c = F.l1_loss(fused, torch.max(img_a, img_b))
    l_g = F.l1_loss(_sobel(fused), torch.max(_sobel(img_a), _sobel(img_b)))
    l_e = l_c + GAMMA * l_g
    l_ssim = (1 - _ssim(fused, img_a)) + (1 - _ssim(fused, img_b))
    return l_e + ALPHA * l_ssim, l_c, l_g, l_ssim


def train_step(state, batch, args, epoch):
    vis, ir = batch["func"], batch["mri"]
    fused = state.model(vis, ir)
    loss, l_c, l_g, l_ssim = loss_fn(fused, ir, vis)

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "content": l_c.item(), "grad": l_g.item(), "ssim": l_ssim.item()}


@torch.no_grad()
def fuse(state, batch, args):
    was_training = state.model.training
    state.model.train(True)            # test.py runs the network in training mode
    fused = state.model(batch["func"], batch["mri"])
    state.model.train(was_training)
    lo = fused.amin(dim=(1, 2, 3), keepdim=True)
    hi = fused.amax(dim=(1, 2, 3), keepdim=True)
    return (fused - lo) / (hi - lo).clamp_min(1e-8)


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
