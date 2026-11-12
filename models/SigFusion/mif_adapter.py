"""SigFusion (AAAI 2026) adapter for the shared MIF benchmark (mifkit).

The release (``fengjiawei123/SigFusion``) has the network (``net.py``: ``Signal_Encoder``,
``Signal_Fusion_Decoder`` -- SigFormer with its EWT / inverse-EWT blocks), loss helpers
(``utils/loss.py``) and test scripts -- no training code for either stage.  Training is
written from the paper (``papers_MIF/SIGFusion-AIII.pdf``):

* stage II (fine-tuning on real pairs, Eq. 16) with the MMIF loss of Eq. (2):
      L = alpha1 * L_mse + beta1 * L_ssim + gamma1 * (L_max + L_grad),
      alpha1 = 2, beta1 = 1, gamma1 = 10;
  the sources serve as supervision (Eq. 16) -- the paper names but does not define the
  terms; taken as L_mse = MSE(F, A) + MSE(F, B), L_ssim = (1 - SSIM(F, A)) + (1 - SSIM(F, B))
  (Gaussian 11x11 SSIM), and L_max, L_grad = the release's ``utils/loss.Fusionloss`` terms:
  L1(F, max(A, B)) and L1(|grad F|, max(|grad A|, |grad B|)) with its Sobel operator;
* Adam(lr 1e-4) (paper; 100 epochs at batch 16 replaced by the benchmark protocol).

Not reproduced -- stage I (pre-training SigFormer on PLGN-synthesised pairs from natural
images): neither PLGN nor its pre-trained weights are released (only the task-fine-tuned
checkpoints, whose MIF model was trained on Harvard images that overlap the benchmark's
test folds); SigFormer is trained from scratch on each fold.

Inputs and output as ``test_MIF.py``: IR = MRI (Y), VIS = the functional image's Y;
``Fusion_Decoder(Encoder(IR), Encoder(VIS))``, a sigmoid output in [0,1] = the fused Y.
``net.py`` moves tensors with ``.cuda()`` (one GPU per job: cuda:0).
"""

import kornia
import torch
import torch.nn.functional as F

from net import Signal_Encoder, Signal_Fusion_Decoder
from utils.loss import Fusionloss

NAME = "SigFusion"
# full-resolution SigFormer: at 256x256 batch 32 and above run out of memory on an 80 GB A100;
# 24 peaks at 74.0 GB (paper: 16)
DEFAULTS = {"batch_size": 24, "lr": 1e-4}
ALPHA1, BETA1, GAMMA1 = 2.0, 1.0, 10.0
HPARAMS = {
    "optimizer": "Adam(lr=1e-4) -- paper (no training code released)",
    "loss": "2*(MSE(F,A)+MSE(F,B)) + 1*((1-SSIM(F,A))+(1-SSIM(F,B))) + 10*(L1(F,max) + L1(grad F, max grad)) "
            "-- paper Eq. (2) weights; term definitions are the benchmark's reading (not defined in the paper)",
    "not_reproduced": "stage I PLGN pre-training (not released); trained from scratch per fold",
    "inputs": "IR = MRI, VIS = func Y -- test_MIF.py",
    "code_defaults_not_used": "100 epochs, batch 16 (benchmark protocol instead)",
}


class State:
    def __init__(self, encoder, decoder, optimizer, fusionloss):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.fusionloss = fusionloss


def build(args, device, pair):
    encoder = Signal_Encoder().to(device)
    decoder = Signal_Fusion_Decoder().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    return State(encoder, decoder, optimizer, Fusionloss(device=device))


def describe(state):
    n = sum(p.numel() for m in (state.encoder, state.decoder) for p in m.parameters())
    return f"SigFusion (SigFormer): {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.encoder.train(train)
    state.decoder.train(train)


def _fuse(state, ir, vis):
    feature_ir, ewt_ir, mfb_ir = state.encoder(ir)
    feature_vis, ewt_vis, mfb_vis = state.encoder(vis)
    return state.decoder(feature_ir, ewt_ir, mfb_ir, feature_vis, ewt_vis, mfb_vis)


def _ssim(a, b):
    return kornia.metrics.ssim(a, b, window_size=11, max_val=1.0).mean()


def train_step(state, batch, args, epoch):
    ir, vis = batch["mri"], batch["func"]
    fused = _fuse(state, ir, vis)
    l_mse = F.mse_loss(fused, ir) + F.mse_loss(fused, vis)
    l_ssim = (1 - _ssim(fused, ir)) + (1 - _ssim(fused, vis))
    _, l_max, l_grad = state.fusionloss(vis, ir, fused)
    loss = ALPHA1 * l_mse + BETA1 * l_ssim + GAMMA1 * (l_max + l_grad)

    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    return {"loss": loss.item(), "mse": l_mse.item(), "ssim": l_ssim.item(), "max": l_max.item(),
            "grad": l_grad.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return _fuse(state, batch["mri"], batch["func"])


def state_dict(state):
    return {"encoder": state.encoder.state_dict(), "decoder": state.decoder.state_dict(),
            "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.encoder.load_state_dict(blob["encoder"])
    state.decoder.load_state_dict(blob["decoder"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
